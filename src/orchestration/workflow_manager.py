"""
Workflow management with DynamoDB integration for tracking and audit logging.
"""
import uuid
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError

from config.aws_config import config
from src.data_models.workflow_legacy import WorkflowManifest
from src.utils.retry_utils import retry_with_exponential_backoff


class WorkflowManager:
    """Manages workflow state, persistence, and audit logging."""

    def __init__(
        self,
        dynamodb_client: Optional[Any] = None,
        s3_client: Optional[Any] = None,
        logs_client: Optional[Any] = None
    ):
        """
        Initialize WorkflowManager.

        Args:
            dynamodb_client: Optional boto3 DynamoDB client (testing)
            s3_client: Optional boto3 S3 client (for testing)
            logs_client: Optional boto3 CloudWatch Logs client (testing)
        """
        session_kwargs = config.get_boto3_session_kwargs()

        self.dynamodb = (
            dynamodb_client or boto3.client('dynamodb', **session_kwargs)
        )
        self.s3 = s3_client or boto3.client('s3', **session_kwargs)
        self.logs = logs_client or boto3.client('logs', **session_kwargs)
        self.workflows_table = config.workflows_table
        self.artifacts_bucket = config.artifacts_bucket
        self.log_group = config.log_group

    def create_workflow(
        self,
        workflow_type: str,
        configuration: Dict[str, Any],
        created_by: str = "system"
    ) -> str:
        """
        Create new workflow instance.

        Args:
            workflow_type: Type (baseline, comparative, fine-tuning)
            configuration: Workflow configuration parameters
            created_by: User or system that created the workflow

        Returns:
            Unique workflow identifier
        """
        # Generate unique workflow ID
        workflow_id = self._generate_workflow_id(workflow_type)

        # Calculate dataset checksum if dataset_s3_uri is provided
        dataset_s3_uri = configuration.get('dataset_s3_uri', '')
        dataset_checksum = (
            self._calculate_checksum(dataset_s3_uri)
            if dataset_s3_uri else ''
        )

        # Extract model IDs from configuration
        model_ids = self._extract_model_ids(configuration)

        # Create workflow manifest
        manifest = WorkflowManifest(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            status='created',
            configuration=configuration,
            dataset_s3_uri=dataset_s3_uri,
            dataset_checksum=dataset_checksum,
            model_ids=model_ids,
            results_s3_uri=None,
            created_at=datetime.utcnow(),
            completed_at=None,
            created_by=created_by,
            events=[]
        )

        # Store manifest in DynamoDB
        self._store_manifest_dynamodb(manifest)

        # Store manifest in S3 for long-term storage
        self._store_manifest_s3(manifest)

        # Log workflow creation event
        self._log_event(
            workflow_id,
            'workflow_created',
            {
                'workflow_type': workflow_type,
                'created_by': created_by,
                'configuration': configuration
            }
        )

        return workflow_id

    def update_workflow_status(
        self,
        workflow_id: str,
        status: str,
        details: Optional[Dict] = None
    ) -> None:
        """
        Update workflow status.
        
        Args:
            workflow_id: Workflow identifier
            status: New status (running, completed, failed)
            details: Optional status details
        """
        # Get current manifest
        manifest = self._get_manifest_dynamodb(workflow_id)
        if not manifest:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Update status
        manifest.status = status
        
        # Update completed_at if status is completed or failed
        if status in ['completed', 'failed']:
            manifest.completed_at = datetime.utcnow()
        
        # Add event to manifest
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'status_update',
            'status': status,
            'details': details or {}
        }
        manifest.events.append(event)
        
        # Update results_s3_uri if provided in details
        if details and 'results_s3_uri' in details:
            manifest.results_s3_uri = details['results_s3_uri']
        
        # Store updated manifest
        self._store_manifest_dynamodb(manifest)
        self._store_manifest_s3(manifest)
        
        # Log status update event
        self._log_event(
            workflow_id,
            'status_update',
            {
                'status': status,
                'details': details or {}
            }
        )
    
    def get_workflow_history(
        self,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve workflow history.
        
        Args:
            filters: Optional filters (date_range, workflow_type, status)
            
        Returns:
            List of workflow summaries
        """
        filters = filters or {}
        
        # Scan DynamoDB table (in production, use indexes for better performance)
        try:
            response = self.dynamodb.scan(
                TableName=self.workflows_table
            )
            
            items = response.get('Items', [])
            
            # Convert DynamoDB items to workflow summaries
            workflows = []
            for item in items:
                workflow = self._dynamodb_item_to_dict(item)
                
                # Apply filters
                if self._matches_filters(workflow, filters):
                    workflows.append(workflow)
            
            # Sort by created_at descending
            workflows.sort(
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )
            
            return workflows
            
        except ClientError as e:
            raise RuntimeError(f"Failed to retrieve workflow history: {e}")
    
    def reproduce_workflow(
        self,
        workflow_id: str,
        created_by: str = "system"
    ) -> str:
        """
        Reproduce a previous workflow with identical configuration.
        
        Args:
            workflow_id: Original workflow identifier
            created_by: User or system reproducing the workflow
            
        Returns:
            New workflow identifier for reproduction
        """
        # Get original manifest
        original_manifest = self._get_manifest_dynamodb(workflow_id)
        if not original_manifest:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create new workflow with same configuration
        new_workflow_id = self.create_workflow(
            workflow_type=original_manifest.workflow_type,
            configuration=original_manifest.configuration,
            created_by=created_by
        )
        
        # Log reproduction event
        self._log_event(
            new_workflow_id,
            'workflow_reproduced',
            {
                'original_workflow_id': workflow_id,
                'reproduced_by': created_by
            }
        )
        
        return new_workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowManifest]:
        """
        Get workflow manifest by ID.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            WorkflowManifest or None if not found
        """
        return self._get_manifest_dynamodb(workflow_id)
    
    def _generate_workflow_id(self, workflow_type: str) -> str:
        """Generate unique workflow identifier."""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"{workflow_type}-{timestamp}-{unique_id}"
    
    def _calculate_checksum(self, s3_uri: str) -> str:
        """
        Calculate checksum for S3 object.
        
        Args:
            s3_uri: S3 URI (s3://bucket/key)
            
        Returns:
            SHA256 checksum
        """
        if not s3_uri or not s3_uri.startswith('s3://'):
            return ''
        
        try:
            # Parse S3 URI
            parts = s3_uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            # Get object
            response = self.s3.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            
            # Calculate SHA256
            return hashlib.sha256(content).hexdigest()
            
        except Exception:
            # Return empty string if checksum calculation fails
            return ''
    
    def _extract_model_ids(self, configuration: Dict[str, Any]) -> List[str]:
        """Extract model IDs from configuration."""
        model_ids = []
        
        # Check common configuration keys
        if 'model_id' in configuration:
            model_ids.append(configuration['model_id'])
        if 'baseline_model_id' in configuration:
            model_ids.append(configuration['baseline_model_id'])
        if 'finetuned_model_id' in configuration:
            model_ids.append(configuration['finetuned_model_id'])
        if 'base_model_id' in configuration:
            model_ids.append(configuration['base_model_id'])
        
        return list(set(model_ids))  # Remove duplicates
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def _store_manifest_dynamodb(self, manifest: WorkflowManifest) -> None:
        """Store workflow manifest in DynamoDB."""
        try:
            item = self._manifest_to_dynamodb_item(manifest)
            self.dynamodb.put_item(
                TableName=self.workflows_table,
                Item=item
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to store manifest in DynamoDB: {e}")
    
    def _store_manifest_s3(self, manifest: WorkflowManifest) -> None:
        """Store workflow manifest in S3 for long-term storage."""
        try:
            key = f"workflows/{manifest.workflow_id}/manifest.json"
            self.s3.put_object(
                Bucket=self.artifacts_bucket,
                Key=key,
                Body=manifest.to_json(),
                ContentType='application/json'
            )
        except ClientError as e:
            # Log error but don't fail - DynamoDB is primary storage
            print(f"Warning: Failed to store manifest in S3: {e}")
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
    def _get_manifest_dynamodb(self, workflow_id: str) -> Optional[WorkflowManifest]:
        """Retrieve workflow manifest from DynamoDB."""
        try:
            response = self.dynamodb.get_item(
                TableName=self.workflows_table,
                Key={'workflow_id': {'S': workflow_id}}
            )
            
            if 'Item' not in response:
                return None
            
            item_dict = self._dynamodb_item_to_dict(response['Item'])
            return WorkflowManifest.from_dict(item_dict)
            
        except ClientError as e:
            raise RuntimeError(f"Failed to retrieve manifest from DynamoDB: {e}")
    
    def _manifest_to_dynamodb_item(
        self, manifest: WorkflowManifest
    ) -> Dict[str, Any]:
        """Convert WorkflowManifest to DynamoDB item format."""
        return {
            'workflow_id': {'S': manifest.workflow_id},
            'workflow_type': {'S': manifest.workflow_type},
            'status': {'S': manifest.status},
            'configuration': {'S': json.dumps(manifest.configuration)},
            'dataset_s3_uri': {'S': manifest.dataset_s3_uri},
            'dataset_checksum': {'S': manifest.dataset_checksum},
            'model_ids': {
                'L': [{'S': mid} for mid in manifest.model_ids]
            },
            'results_s3_uri': {'S': manifest.results_s3_uri or ''},
            'created_at': {'S': manifest.created_at.isoformat()},
            'completed_at': {
                'S': (
                    manifest.completed_at.isoformat()
                    if manifest.completed_at else ''
                )
            },
            'created_by': {'S': manifest.created_by},
            'events': {'S': json.dumps(manifest.events)}
        }
    
    def _dynamodb_item_to_dict(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DynamoDB item to dictionary."""
        return {
            'workflow_id': item['workflow_id']['S'],
            'workflow_type': item['workflow_type']['S'],
            'status': item['status']['S'],
            'configuration': json.loads(item['configuration']['S']),
            'dataset_s3_uri': item['dataset_s3_uri']['S'],
            'dataset_checksum': item['dataset_checksum']['S'],
            'model_ids': [mid['S'] for mid in item['model_ids']['L']],
            'results_s3_uri': item['results_s3_uri']['S'] or None,
            'created_at': item['created_at']['S'],
            'completed_at': item['completed_at']['S'] or None,
            'created_by': item['created_by']['S'],
            'events': json.loads(item['events']['S'])
        }
    
    def _matches_filters(
        self, workflow: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """Check if workflow matches filter criteria."""
        # Filter by workflow_type
        if 'workflow_type' in filters:
            if workflow.get('workflow_type') != filters['workflow_type']:
                return False
        
        # Filter by status
        if 'status' in filters:
            if workflow.get('status') != filters['status']:
                return False
        
        # Filter by date_range
        if 'date_range' in filters:
            date_range = filters['date_range']
            created_at = workflow.get('created_at', '')
            
            if 'start' in date_range:
                if created_at < date_range['start']:
                    return False
            
            if 'end' in date_range:
                if created_at > date_range['end']:
                    return False
        
        return True
    
    def _log_event(
        self,
        workflow_id: str,
        event_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log workflow event to CloudWatch.
        
        Args:
            workflow_id: Workflow identifier
            event_type: Type of event
            details: Event details
        """
        try:
            # Ensure log group exists
            self._ensure_log_group_exists()
            
            # Create log stream if it doesn't exist
            log_stream = f"workflow-{workflow_id}"
            self._ensure_log_stream_exists(log_stream)
            
            # Prepare log message
            log_message = json.dumps({
                'timestamp': datetime.utcnow().isoformat(),
                'workflow_id': workflow_id,
                'event_type': event_type,
                'details': details
            })
            
            # Put log event
            self.logs.put_log_events(
                logGroupName=self.log_group,
                logStreamName=log_stream,
                logEvents=[
                    {
                        'timestamp': int(datetime.utcnow().timestamp() * 1000),
                        'message': log_message
                    }
                ]
            )
            
        except Exception as e:
            # Log error but don't fail workflow
            print(f"Warning: Failed to log event to CloudWatch: {e}")
    
    def _ensure_log_group_exists(self) -> None:
        """Ensure CloudWatch log group exists."""
        try:
            self.logs.create_log_group(logGroupName=self.log_group)
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            print(f"Warning: Failed to create log group: {e}")
    
    def _ensure_log_stream_exists(self, log_stream: str) -> None:
        """Ensure CloudWatch log stream exists."""
        try:
            self.logs.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=log_stream
            )
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            print(f"Warning: Failed to create log stream: {e}")

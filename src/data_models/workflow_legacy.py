"""
Legacy dataclass WorkflowManifest for backward compatibility.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class WorkflowManifest:
    """Legacy workflow manifest with JSON serialization."""
    workflow_id: str
    workflow_type: str
    status: str
    configuration: dict = field(default_factory=dict)
    dataset_s3_uri: str = ""
    dataset_checksum: str = ""
    model_ids: list = field(default_factory=list)
    results_s3_uri: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = ""
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'workflow_id': self.workflow_id,
            'workflow_type': self.workflow_type,
            'status': self.status,
            'configuration': self.configuration,
            'dataset_s3_uri': self.dataset_s3_uri,
            'dataset_checksum': self.dataset_checksum,
            'model_ids': self.model_ids,
            'results_s3_uri': self.results_s3_uri,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_by': self.created_by,
            'events': self.events,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorkflowManifest':
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        completed_at = data.get('completed_at')
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        return cls(
            workflow_id=data['workflow_id'],
            workflow_type=data['workflow_type'],
            status=data['status'],
            configuration=data.get('configuration', {}),
            dataset_s3_uri=data.get('dataset_s3_uri', ''),
            dataset_checksum=data.get('dataset_checksum', ''),
            model_ids=data.get('model_ids', []),
            results_s3_uri=data.get('results_s3_uri'),
            created_at=created_at,
            completed_at=completed_at,
            created_by=data.get('created_by', ''),
            events=data.get('events', []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'WorkflowManifest':
        return cls.from_dict(json.loads(json_str))

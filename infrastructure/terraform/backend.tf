# Terraform Backend Configuration
#
# Remote state storage for team collaboration and disaster recovery.
#
# The backend is deliberately a partial configuration: no bucket, key, region,
# or lock table is hardcoded here. S3 bucket names are globally unique across
# all of AWS, so committing a short predictable name would let another account
# create it first and receive your state file, which contains resource
# identifiers and can contain secrets.
#
# Supply the values at init time from a file that is not committed:
#
#   terraform init -backend-config=backend.hcl
#
# backend.hcl (gitignored; see backend.hcl.example):
#
#   bucket         = "trustops-terraform-state-<ACCOUNT_ID>-<REGION>"
#   key            = "trustops/terraform.tfstate"
#   region         = "<REGION>"
#   encrypt        = true
#   dynamodb_table = "trustops-terraform-locks"
#
# Create the prerequisites first, substituting your own account ID and region:
#
#   export AWS_REGION=us-east-1
#   STATE_BUCKET="trustops-terraform-state-$(aws sts get-caller-identity \
#       --query Account --output text)-${AWS_REGION}"
#   aws s3 mb "s3://${STATE_BUCKET}" --region "${AWS_REGION}"
#   aws s3api put-bucket-versioning --bucket "${STATE_BUCKET}" \
#       --versioning-configuration Status=Enabled
#   aws s3api put-bucket-encryption --bucket "${STATE_BUCKET}" \
#       --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
#   aws s3api put-public-access-block --bucket "${STATE_BUCKET}" \
#       --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
#   aws dynamodb create-table --table-name trustops-terraform-locks \
#       --attribute-definitions AttributeName=LockID,AttributeType=S \
#       --key-schema AttributeName=LockID,KeyType=HASH \
#       --billing-mode PAY_PER_REQUEST --region "${AWS_REGION}"
#
# Also apply a DenyInsecureTransport bucket policy to the state bucket, the
# same control the framework's own buckets carry.

terraform {
  # Partial configuration: every value is supplied via -backend-config.
  backend "s3" {
  }
}

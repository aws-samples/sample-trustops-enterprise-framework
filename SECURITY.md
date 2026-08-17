# Security Policy

## Reporting a Vulnerability

If you discover a potential security issue in this project, please notify
AWS/Amazon Security via our
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).
Please do **not** create a public GitHub issue.

## Non-Production Disclaimer

This project is provided as a sample and educational implementation. It
demonstrates how to build an evaluation and trust-gating workflow on Amazon
Bedrock; it is **not** a production-ready service.

It is **NOT intended for production use** without the additional hardening
described below. In particular, the deployed infrastructure makes trade-offs
that favour ease of first deployment over defence in depth: Lambda functions
run outside a VPC, IAM policies are scoped to the sample's own resources
rather than to a reviewed production boundary, and no data-classification or
PII-handling controls are applied to the datasets you supply.

Before running this against real workloads or customer data, work through the
hardening checklist and the known security debt table with your own security
reviewers.

## Production Hardening

The following changes are recommended before any production use. None are
applied by the sample templates.

### Network isolation

- Place all Lambda functions in a VPC with private subnets, and reach AWS
  services through VPC endpoints (`bedrock-runtime`, `s3`, `dynamodb`,
  `logs`, `sts`) rather than public endpoints.
- Attach security groups that deny egress by default and allow only the
  endpoints the function needs.
- Add an S3 bucket policy condition on `aws:SourceVpce` so the data buckets
  are reachable only from your VPC endpoints.

### Identity and access management

- Replace the sample execution roles with least-privilege roles scoped to
  named resources. Remove any `Resource: "*"` grant that your own audit does
  not accept (see SD-1).
- Restrict the KMS key policy to the specific roles that need each operation
  instead of relying on the account-root delegation pattern (see SD-2).
- Require MFA for human principals that can read the results buckets.
- Enable IAM Access Analyzer and review external-access findings on the
  buckets, tables, and the KMS key.

### Data protection

- Classify the datasets you load. This framework logs model prompts and
  responses to CloudWatch Logs and S3; if those contain PII or regulated
  data, add redaction before logging and shorten retention accordingly.
- Enable Amazon Macie on the datasets and results buckets to detect
  sensitive data that arrives unexpectedly.
- Enable S3 Object Lock on the results bucket if evaluation results are used
  as audit evidence and must be immutable.
- Enable AWS Backup for the DynamoDB tables in addition to point-in-time
  recovery.

### Responsible AI

- Attach Amazon Bedrock Guardrails to every model invocation to filter
  harmful content and block prompt-injection attempts.
- Do not use model output as a sole decision-maker in any consequential
  domain. The medical, financial, and legal dataset templates shipped here
  carry explicit warnings to this effect; treat them as illustrative
  structure only, not as validated evaluation sets for those domains.
- Keep a human review step between a `Deploy` verdict from this framework
  and an actual production model promotion.

### Detection and response

- Enable AWS CloudTrail data events for the S3 buckets and the KMS key.
- Enable Amazon GuardDuty, including S3 and Lambda protection.
- Route the CloudWatch alarms to a monitored on-call channel rather than an
  unsubscribed SNS topic.
- Enable AWS Config with the security conformance pack to catch drift from
  the hardened baseline.

### Supply chain

- Pin every dependency to an exact version and generate a lock file, then
  patch on a deliberate schedule with a CI gate (see SD-6).
- Pin container base images by digest and rebuild on a schedule to pick up
  base-image CVE fixes.
- Run a software composition analysis scan in CI and fail the build on new
  high-severity CVEs.

## Known Security Debt

These items are accepted trade-offs in the sample. Each has a rationale and a
production remediation. They are not defects, but they are also not choices
you should inherit unexamined.

| # | Item | Severity | Justification | Production remediation |
|---|---|---|---|---|
| SD-1 | IAM `Resource: "*"` for CloudWatch Logs and X-Ray | MEDIUM | These APIs do not support resource-level permissions, so `"*"` is the only valid value. Scope is constrained by condition keys where the API allows it. | Keep `"*"` where required by the API, but add `aws:SourceAccount`/namespace conditions and confirm via IAM Access Analyzer policy validation. |
| SD-2 | KMS key policy delegates `kms:*` to the account root | MEDIUM | Standard AWS key policy pattern. The root delegation is what allows IAM policies to govern key use and prevents the key from becoming unmanageable. | Enumerate the specific roles that need `Encrypt`, `Decrypt`, and `GenerateDataKey`, and reduce the root statement to key administration only. |
| SD-3 | Lambda functions are not in a VPC | LOW | The functions reach only AWS managed services over public service endpoints; no private resources are accessed. | Move functions into private subnets with VPC endpoints, as described under Network isolation. |
| SD-4 | Access-log bucket has no server-access logging of its own | LOW | Logging a log bucket to itself creates infinite recursion, and logging it elsewhere creates a circular dependency. This is a documented AWS exception. | Leave as is; rely on CloudTrail data events for audit of the log bucket. |
| SD-5 | Region is hardcoded as a Terraform variable default | MEDIUM | The default is a convenience for first deployment and is overridable via `terraform.tfvars` or `-var`. | Remove the default so the region must be stated explicitly per environment. |
| SD-6 | Python dependencies use open version ranges (`>=`) | MEDIUM | Open ranges pick up CVE patches automatically. The trade-off is reproducibility against patch latency; for sample code, auto-patching is the safer default. | Pin exact versions, commit a lock file, and move patching behind a CI gate with SCA scanning. |
| SD-7 | OpenSearch Serverless network policy sets `AllowFromPublic = true` | HIGH | The collection is reachable over the public endpoint, but every request is still authenticated and authorized by IAM through a separate data access policy; there is no anonymous access. Restricting the collection to a VPC is an architectural change requiring VPC endpoints and Lambda VPC attachment, which this sample does not provision. | Create an `aoss` VPC endpoint, set `AllowFromPublic = false` with `SourceVPCEs` on the network policy, and move the Lambda functions into private subnets (see SD-3 and Network isolation). |
| SD-8 | Fictional phone number `555-123-4567` sits outside the NANP reserved range | LOW | Appears only in PII-detector and safety-scorer test fixtures and their documentation, never in deployed configuration. It is test data for the redaction path, not a contactable number. | Move sample numbers into the NANP-reserved `555-0100`–`555-0199` block so no fixture can resolve to a real subscriber. |

## Security Scanning

This repository is scanned with the AWS Automated Security Helper (ASH) and
its bundled tools. Current state:

| Scanner | Result |
|---|---|
| bandit | 0 findings; suppressions are rule-specific and carry an inline reason |
| checkov (Terraform) | 0 failures; every skip is documented inline |
| checkov (CloudFormation) | 0 failures; every skip is documented inline |
| cfn-lint | 0 findings |
| semgrep (`p/security-audit`, `p/secrets`) | 0 findings |
| detect-secrets | Baselined in `.secrets.baseline`; all entries reviewed as documentation placeholders or mock test credentials |

If you add a suppression, include the rule ID and the reason it does not
apply. A suppression without a stated rationale should not pass review.

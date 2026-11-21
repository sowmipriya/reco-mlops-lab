variable "project_id" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  description = "Default region for regional resources"
  type        = string
  default     = "europe-west1"
}

variable "bq_location" {
  description = "BigQuery / GCS multi-region"
  type        = string
  default     = "EU"
}

variable "env" {
  description = "Environment label (dev, prd)"
  type        = string
  default     = "dev"
}

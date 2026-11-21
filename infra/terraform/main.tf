terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  labels = {
    app        = "reco"
    managed_by = "terraform"
    env        = var.env
  }

  # Medallion-style layers, one BigQuery dataset each.
  datasets = {
    raw        = "Raw data landed from source domains (events, catalog, customers, promotions)"
    staging    = "dbt staging models: typed, renamed, deduplicated"
    marts      = "dbt marts: dimensions and facts"
    features   = "dbt feature tables consumed by training and the online store"
    elementary = "Elementary data observability results (dbt test history, anomalies)"
  }
}

resource "google_project_service" "services" {
  for_each           = toset(["bigquery.googleapis.com", "storage.googleapis.com", "iam.googleapis.com"])
  service            = each.value
  disable_on_destroy = false
}

resource "google_bigquery_dataset" "layers" {
  for_each    = local.datasets
  dataset_id  = each.key
  description = each.value
  location    = var.bq_location
  labels      = local.labels

  # Keep raw data cheap: expire partitions after 400 days by default.
  default_partition_expiration_ms = each.key == "raw" ? 400 * 24 * 3600 * 1000 : null

  depends_on = [google_project_service.services]
}

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-artifacts"
  location                    = var.bq_location
  uniform_bucket_level_access = true
  labels                      = local.labels

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

# Identity used by scheduled pipelines (dbt, training, materialization).
resource "google_service_account" "pipelines" {
  account_id   = "reco-pipelines"
  display_name = "Reco data & ML pipelines"
}

resource "google_project_iam_member" "pipelines_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.pipelines.email}"
}

resource "google_bigquery_dataset_iam_member" "pipelines_editor" {
  for_each   = google_bigquery_dataset.layers
  dataset_id = each.value.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.pipelines.email}"
}

resource "google_storage_bucket_iam_member" "pipelines_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipelines.email}"
}

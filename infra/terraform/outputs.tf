output "datasets" {
  value = [for d in google_bigquery_dataset.layers : d.dataset_id]
}

output "artifacts_bucket" {
  value = google_storage_bucket.artifacts.url
}

output "pipelines_service_account" {
  value = google_service_account.pipelines.email
}

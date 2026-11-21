{#- Use the custom schema as-is so layers map 1:1 to the BigQuery datasets
    created by Terraform (raw, staging, marts, features). -#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}{{ target.schema }}{%- else -%}{{ custom_schema_name | trim }}{%- endif -%}
{%- endmacro %}

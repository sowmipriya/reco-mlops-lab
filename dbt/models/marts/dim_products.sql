select
    product_id,
    product_name,
    category,
    subcategory,
    category_code,
    brand,
    price,
    is_active
from {{ ref('stg_products') }}

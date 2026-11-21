select
    product_id,
    product_name,
    category,
    subcategory,
    brand,
    cast(price_eur as numeric) as price_eur,
    is_active
from {{ source('raw', 'products') }}

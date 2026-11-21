select
    product_id,
    product_name,
    category,
    subcategory,
    category_code,
    nullif(brand, 'unbranded')  as brand,
    cast(price as numeric)      as price,
    -- A product without a price cannot be sold, so it is never recommended.
    is_active and price > 0     as is_active
from {{ source('raw', 'products') }}

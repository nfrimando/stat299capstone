with
grouped as (
    select      po.buyer_id,
                count(*) as count
    from        purchase_orders po
                left join purchase_order_documents pod On pod.purchase_order_id = po.id
                left join purchase_order_attachments poa ON poa.purchase_order_document_id = pod.id
    where       1 = 1
                -- and po.buyer_id = 9211
                -- and po.buyer_id = 9763
                -- and po.buyer_id = 9592
                -- and po.buyer_id = 7801
                -- and po.buyer_id = 12754
                -- and po.buyer_id = 66035
                and poa.attachment_file_content_type = 'application/pdf'
                and po.buyer_id is not null
    group by    1
    order by    2 desc
),

prelim as (

    select      *,
                row_number() over (order by count desc) as order_sum
    from        grouped
)

select      p.*,
            sum(p.count) over (order by order_sum),
            b.name
from        prelim p
            left join buyers b on b.id = p.buyer_id

select      po.id,
            po.buyer_id,
            po.loan_application_id,
            poa.id as attachment_id,
            'https://s3.console.aws.amazon.com/s3/buckets/firstcircle/production/loan_applications/' || po.loan_application_id || '/purchase_order_attachments/' || poa.id || '/?region=ap-southeast-1&tab=overview' as link
from        purchase_orders po
            left join purchase_order_documents pod On pod.purchase_order_id = po.id
            left join purchase_order_attachments poa ON poa.purchase_order_document_id = pod.id
where       1 = 1
            -- and po.buyer_id = 12147
            and poa.attachment_file_content_type = 'application/pdf'

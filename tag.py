import json
import os
import base64
import boto3
import tempfile
from birds_detection import image_prediction
from boto3.dynamodb.conditions import Attr

# 環境變數
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "BirdMediaMetadata")
table = boto3.resource("dynamodb").Table(TABLE_NAME)

def lambda_handler(event, context):
    print("DEBUG: Raw event =", json.dumps(event))

    raw_body = event.get("body")
    if not raw_body:
        return _response(400, {"error": "Missing request body"})

    try:
        body = json.loads(raw_body)
    except Exception as e:
        return _response(400, {"error": "Invalid JSON format"})

    file_b64 = body.get("file_content_base64")
    filename = body.get("filename", "uploaded.jpg")

    if not file_b64 or not isinstance(file_b64, str):
        return _response(400, {"error": "Missing or invalid 'file_content_base64'"})

    # 1. Decode base64 → bytes
    try:
        file_bytes = base64.b64decode(file_b64)
    except Exception as e:
        return _response(400, {"error": f"Base64 decode failed: {str(e)}"})

    # 2. 寫入暫存檔並跑模型推論
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = image_prediction(
            image_path=tmp_path,
            result_filename=None,
            save_dir="/tmp",
            confidence=0.5,
            model="./model.pt"
        )
    except Exception as e:
        return _response(500, {"error": f"Model inference error: {str(e)}"})

    tags = result.get("tags", [])
    if not tags:
        return _response(200, {"tags": [], "links": []})

    print("DEBUG: Inferred tags =", tags)

    # 3. 用 tags 去查詢 DynamoDB
    filter_expr = None
    for idx, tag in enumerate(tags):
        cond = Attr(f"Tags.{tag}").gte(1)
        filter_expr = cond if idx == 0 else filter_expr | cond

    try:
        response = table.scan(FilterExpression=filter_expr)
        items = response.get("Items", [])
    except Exception as e:
        return _response(500, {"error": f"DynamoDB scan error: {str(e)}"})

    result_links = []
    for item in items:
        filetype = item.get("FileType", "image")
        if filetype == "image":
            thumb = item.get("ThumbnailURL")
            if thumb:
                result_links.append(thumb)
        else:
            full = item.get("OriginalURL")
            if full:
                result_links.append(full)

    return _response(200, {
        "tags": tags,
        "links": result_links
    })

def _response(status, body_dict):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_dict)
    }


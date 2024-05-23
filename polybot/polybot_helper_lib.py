import boto3
import logging
import os
from botocore.exceptions import ClientError


def upload_file(file_name, bucket, s3_client, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def count_objects_in_dict(mydict):
    obj_count = {}
    for i in mydict:
        obj_name = mydict.get(i).get('class')

        if obj_count.get(obj_name) is not None:
            obj_count[obj_name] = obj_count[obj_name] + 1
        else:
            obj_count[obj_name] = 1

    return obj_count

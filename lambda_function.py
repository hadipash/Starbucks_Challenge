"""
This function is to be deployed on AWS Lambda to access the endpoints through the Amazon API Gateway
"""

import boto3
import json


def lambda_handler(event, context):
    """
    Sends received through API data for inference to the endpoints and returns back predictions

    :param event: dict. AWS Lambda uses this parameter to pass in event data to the handler
    :param context: AWS Lambda uses this parameter to provide runtime information to the handler
    :return: dict. server response with body containing model's predictions
    """
    endpoints = {
        'bogo': 'sagemaker-xgboost-200201-1515-020-113da15d',
        'disc': 'sagemaker-xgboost-200201-1550-030-eb4c2775',
        'info': 'sagemaker-xgboost-200201-1626-003-f9a40baf'
    }

    # Use the SageMaker runtime to invoke the endpoints
    runtime = boto3.Session().client('sagemaker-runtime')

    responses = {}
    for name, endpoint in endpoints.items():
        response = runtime.invoke_endpoint(EndpointName=endpoint,
                                           ContentType='text/csv',
                                           Body=event['body'])
        responses[name] = response['Body'].read().decode('utf-8')

    return {
        'statusCode': 200,
        'body': json.dumps(responses)
    }

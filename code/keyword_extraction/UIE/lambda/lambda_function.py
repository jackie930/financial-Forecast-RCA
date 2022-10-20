import json
import boto3

# Global variables that are reused
sm_runtime_client = boto3.client('sagemaker-runtime')

def get_result(sm_runtime_client, sagemaker_endpoint, text):
    response = sm_runtime_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint,
        ContentType='application/json',
        Body=json.dumps(text))
    response_body = json.loads((response['Body'].read()))
    result = response_body
    return result

def lambda_handler(event, context):
    # TODO implement
    # sagemaker variables
    # sagemaker_endpoint = 'pytorch-inference-2022-07-05-07-28-16-183'  # m5.2xlarge
    sagemaker_endpoint = 'pytorch-inference-2022-07-06-04-02-11-091'  # g4dn.xlarge
    
    print(event)
    query = json.loads(event['body'])
    text = [query['text']]
    # text = ['北京某某律师事务所']
    result = get_result(sm_runtime_client, sagemaker_endpoint, text)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result, ensure_ascii=False)
    }

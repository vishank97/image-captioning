import yaml
from servicefoundry import Build, PythonBuild, Service, Resources

with open("predict/predict.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Since we are using FastAPI service to infer the model we'll use the Service() method 
service = Service(
    name=env_vars['name'],
    image=Build(
        build_spec=PythonBuild(
            command=env_vars['components'][0]['image']['build_spec']['command'],
        ),
    ),
    ports=[{"port": 8000}],
    resources=Resources(memory_limit=2500, memory_request=2000)
)
service.deploy(workspace_fqn=env_vars['components'][0]['env']['WORKSPACE_FQN'])

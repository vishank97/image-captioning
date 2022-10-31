from servicefoundry import Build, PythonBuild, Resources, Service
import yaml


with open("demo/demo.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# creating a service object and defining all the configurations
# Since we are using Streamlit service to show the demo we'll use the Service() method 
service = Service(
    name=env_vars['name'],
    image=Build(
        build_spec=PythonBuild(
            command=env_vars['components'][0]['image']['build_spec']['command'],
            python_version="3.9",
        ),
    ),
    env={
        # These will automatically map the secret value to the environment variable.
        "MLF_HOST": env_vars['components'][0]['env']['MLF_HOST'],
        "MLF_API_KEY": env_vars['components'][0]['env']['MLF_API_KEY']
    },
    ports=[{"port": 8501}], #In public cloud deployment TrueFoundry exposes port 8501
    resources=Resources(
        cpu_request=1, cpu_limit=1.5, memory_limit=15000, memory_request=10000
    ),
)
service.deploy(workspace_fqn=env_vars['components'][0]['env']['WORKSPACE_FQN'])
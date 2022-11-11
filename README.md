# smart-image-captioning

# Predict caption from images and deploy the ML app on the cloud

## Useful Links

[Taxi Fare Prediction model Deployment Video](https://www.loom.com/share/ecae98fc81754b6ebdb31902e9b4025e)

[Link to Live Demo](https://image-caption-demo-demo-projects.tfy-ctl-euwe1-production.production.truefoundry.com/)

[Colab Notebook](https://drive.google.com/file/d/1WL8cnVmqsWxh9Ok-Ml5axAuxGfnZ1A9S/view?usp=sharing)

Blog with instructions on the run (Coming Soon)

## Description of the problem

The aim is to predict the captions of images using deep learning.

## Model Trained

We used a [pretrained GPT-2 model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) and deployed it as a webapp and a FastAPI endpoint using ServiceFoundry 🚀

## Instructions to deploy on ServiceFoundry

<details>

<summary><b><font size="5">Setting up servicefoundry</font></b></summary>

Install and setup servicefoundry on your computer.

```commandline
pip install servicefoundry

servicefoundry use server https://app.truefoundry.com

servicefoundry login
```

</details>


<details>

<summary><b><font  size="5">Deploying realtime inference</font></b></summary>


1. Change working directory to predict folder

```commandline

cd predict

```

2. Create [workspace](https://docs.truefoundry.com/documentation/deploy/concepts/workspace) and [API key](https://docs.truefoundry.com/documentation/deploy/concepts/secrets) on the TrueFoundry platform
3. Replace the ``MLF_API_KEY`` value `predict.yaml` file with the API Key found in [secrets tab](https://app.truefoundry.com/secrets) of your TrueFoundry account [(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/secrets#how-to-store-secrets-in-truefoundry)
4. Copy the workspace_fqn of the workspace that you want to use from the [workspace tab](https://app.truefoundry.com/workspaces) of TrueFoundry[(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name) and add it in `predict.yaml` file


6. To deploy using python script:

```commandline

python predict_deploy.py

```

To deploy using CLI:

```commandline

servicefoundry deploy --file predict/predict_deploy.yaml

```

7. Click on the dashboard link in the terminal to open the service deployment page with FastAPI EndPoint

</details>

<details>

<summary><b><font  size="5">Querying the deployed model</font></b></summary>

This can done via the [fastapi endpoint](https://image-caption-predict-demo-projects.tfy-ctl-euwe1-production.production.truefoundry.com/docs#/default/predict_predict_post) directly via browser.


</details>

<details>

<summary><b><font  size="5">Deploying Demo </font></b></summary>

Note: It is necessary to deploy live inference model before being able to deploy a demo

1. Create [workspace](https://docs.truefoundry.com/documentation/deploy/concepts/workspace) and [API key](https://docs.truefoundry.com/documentation/deploy/concepts/secrets) on the TrueFoundry platform
2. Replace the ``MLF_API_KEY`` value `demo.yaml` file with the API Key found in [secrets tab](https://app.truefoundry.com/secrets) of your TrueFoundry account [(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/secrets#how-to-store-secrets-in-truefoundry)
3. Copy the workspace_fqn of the workspace that you want to use from the [workspace tab](https://app.truefoundry.com/workspaces) of TrueFoundry and add it in the `demo.yaml` file [(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name)

4. To deploy using python script:

```commandline

python demo/demo_deploy.py

```

To deploy using CLI:

```commandline

servicefoundry deploy --file demo/demo_deploy.yaml

```

5. Click on the dashboard link in the terminal
6. Click on the <b>"Endpoint"</b> link on the dashboard to open the streamlit demo

</details>

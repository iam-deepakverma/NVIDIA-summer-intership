# Day 9

## Using roboflow

### To get workspace details

```python
from roboflow import Roboflow
rf=Roboflow(api_key="your api key here")
print(rf. workspace)

```
### To predict the own trained model from roboflow

```python
from roboflow import Roboflow
rf = Roboflow(api_key="your api key here")
workspace = rf. workspace("url of workspace from the workspace output")
project = workspace.project("project")
model = project.version(1) model
result = model.predict("sample.jpg").json()
print(result)

```


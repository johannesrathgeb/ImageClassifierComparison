import plotly.graph_objects as go
import math
from plotly.subplots import make_subplots

#DT
timeDT = [0.071, 0.546, 1.528, 2.988]
scoreDT = [20.5, 23.31, 23.48, 23.61]

#RF
timeRF = [3.5, 61.249, 137.886, 301.376]
scoreRF = [34.31, 42.33, 44.91, 46.8]

#GB
timeGB = [21.426, 176.7, 456.47, 930.656]
scoreGB = [36.36, 46.55, 49.296, 50.88]

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=[1000, 10000, 25000, 50000], y=timeDT, name="Training Time Decision Tree"),
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(x=[1000, 10000, 25000, 50000], y=timeRF, name="Training Time Random Forest"),
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(x=[1000, 10000, 25000, 50000], y=timeGB, name="Training Time Gradient Boosting Tree"),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(x=[1000, 10000, 25000, 50000], y=scoreDT, name="Accuracy Decision Tree"),
    secondary_y=False,
)

fig.add_trace(
    go.Bar(x=[1000, 10000, 25000, 50000], y=scoreRF, name="Accuracy Random Forest"),
    secondary_y=False,
)
fig.add_trace(
    go.Bar(x=[1000, 10000, 25000, 50000], y=scoreGB, name="Accuracy Gradient Boosting Tree"),
    secondary_y=False,
)

# Add figure title
fig.update_layout(
    title_text="<b>Decision Tree</b> vs <b>Random Forest</b> vs <b>Gradient Boosting Tree</b>"
)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

# Set x-axis title
fig.update_xaxes(title_text="Training Batch length")
fig.update_xaxes(type='category')

# Set y-axes titles
fig.update_yaxes(title_text="Training Time in seconds", secondary_y=True, range=[math.floor(timeDT[0]), math.ceil(timeGB[-1])])
fig.update_yaxes(title_text="Accuracy in %", secondary_y=False, range=[math.floor(scoreDT[0]), math.ceil(scoreGB[-1])])

fig.show()
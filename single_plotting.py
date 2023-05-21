import plotly.graph_objects as go
import math
from plotly.subplots import make_subplots

name = "<b>Decision Tree</b>"
time = [0.071, 0.546, 1.528, 2.988]
score = [20.5, 23.31, 23.48, 23.61]

#name = "<b>Random Forest</b>"
#time = [3.5, 61.249, 137.886, 301.376]
#score = [34.31, 42.33, 44.91, 46.8]

#name = "<b>Gradient Boosting Tree</b>"
#time = [21.426, 176.7, 456.47, 930.656]
#score = [36.36, 46.55, 49.296, 50.88]


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=[1000, 10000, 25000, 50000], y=time, name="Training Time"),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(x=[1000, 10000, 25000, 50000], y=score, name="Accuracy", marker=dict(color='#ffa15a')),
    secondary_y=False,
)

# Add figure title
fig.update_layout(
    title_text=name
)

# Set x-axis title
fig.update_xaxes(title_text="Training Batch length")
fig.update_xaxes(type='category')

# Set y-axes titles
fig.update_yaxes(title_text="Training Time in seconds", secondary_y=True, range=[math.floor(time[0]), math.ceil(time[-1])])
fig.update_yaxes(title_text="Accuracy in %", secondary_y=False, range=[math.floor(score[0]), math.ceil(score[-1])])


fig.show()
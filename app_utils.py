import os
from typing import Optional
from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.decorators import StepConfig
from llama_index.core.workflow.utils import (
    get_steps_from_class,
    get_steps_from_instance,
)
from llama_index.core.workflow.workflow import Workflow
import html

def generate_workflow_visualization(workflow: Workflow) -> str:
    """Generates HTML visualization of all possible flows in the workflow."""
    from pyvis.network import Network
    import json

    net = Network(directed=True, height="750px", width="100%", notebook=False)

    # Add the nodes + edge for stop events
    net.add_node(
        StopEvent.__name__,
        label=StopEvent.__name__,
        color="#FFA07A",
        shape="ellipse",
    )
    net.add_node("_done", label="_done", color="#ADD8E6", shape="box")
    net.add_edge(StopEvent.__name__, "_done")

    # Add nodes from all steps
    steps = get_steps_from_class(workflow)
    if not steps:
        # If no steps are defined in the class, try to get them from the instance
        steps = get_steps_from_instance(workflow)

    step_config: Optional[StepConfig] = None
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        net.add_node(
            step_name, label=step_name, color="#ADD8E6", shape="box"
        )  # Light blue for steps

        for event_type in step_config.accepted_events:
            net.add_node(
                event_type.__name__,
                label=event_type.__name__,
                color="#90EE90" if event_type != StartEvent else "#E27AFF",
                shape="ellipse",
            )  # Light green for events

    # Add edges from all steps
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)

        if step_config is None:
            continue

        for return_type in step_config.return_types:
            if return_type != type(None):
                net.add_edge(step_name, return_type.__name__)

        for event_type in step_config.accepted_events:
            net.add_edge(event_type.__name__, step_name)

    # Convert nodes and edges to JSON strings
    nodes_json = json.dumps(list(net.nodes))
    edges_json = json.dumps(list(net.edges))

    # Generate HTML with proper vis.js configuration
    html_content = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-eOJMYsd53ii+scO/bJGFsiCZc+5NDVN2yr8+0RDqr0Ql0h+rP48ckxlpbzKgwra6" crossorigin="anonymous" />
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js" integrity="sha384-JEW9xMcG8R+pH31jmWH6WWP0WintQrMb4s7ZOdauHnUtxwoG2vI5DkLtS3qm9Ekf" crossorigin="anonymous"></script>
            <style type="text/css">
                #mynetwork {{
                    width: 100%;
                    height: 750px;
                    background-color: #ffffff;
                    border: 1px solid lightgray;
                    position: relative;
                    float: left;
                }}
            </style>
        </head>
        <body>
            <div class="card" style="width: 100%">
                <div id="mynetwork" class="card-body"></div>
            </div>
            <script type="text/javascript">
                // Initialize global variables
                var edges;
                var nodes;
                var network;
                var container;

                // This method is responsible for drawing the graph
                function drawGraph() {{
                    var container = document.getElementById('mynetwork');

                    // Parse nodes and edges
                    nodes = new vis.DataSet({nodes_json});
                    edges = new vis.DataSet({edges_json});

                    // Create the network
                    var data = {{
                        nodes: nodes,
                        edges: edges
                    }};

                    var options = {{
                        "configure": {{
                            "enabled": false
                        }},
                        "edges": {{
                            "color": {{
                                "inherit": true
                            }},
                            "smooth": {{
                                "enabled": true,
                                "type": "dynamic"
                            }}
                        }},
                        "interaction": {{
                            "dragNodes": true,
                            "hideEdgesOnDrag": false,
                            "hideNodesOnDrag": false
                        }},
                        "physics": {{
                            "enabled": true,
                            "stabilization": {{
                                "enabled": true,
                                "fit": true,
                                "iterations": 1000,
                                "onlyDynamicEdges": false,
                                "updateInterval": 50
                            }}
                        }}
                    }};

                    network = new vis.Network(container, data, options);
                    return network;
                }}

                drawGraph();
            </script>
        </body>
    </html>
    """
    
    # Escape the HTML content for safe embedding in srcdoc
    return html.escape(html_content)

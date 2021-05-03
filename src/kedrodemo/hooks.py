# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal

import statsd
import sys

import mlflow
import mlflow.sklearn
from kedro.pipeline.node import Node

class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the same run_id as the Kedro pipeline run.
        """
        mlflow.start_run(run_name=run_params["run_id"])
        mlflow.log_params(run_params)

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> None:
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:
        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.
        """
        if node._func_name == "split_data":
            mlflow.log_params(
                # {"split_data_ratio": inputs["params:test_size"]}
                {"split_data_ratio": .20}
            )

        elif node._func_name == "train_model":
            model = outputs["regressor"]
            mlflow.sklearn.log_model(model, "model")
            #mlflow.log_params(inputs["parameters"])

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        mlflow.end_run()
        
        
class PipelineMonitoringHooks:
    def __init__(self):
        self._timers = {}
        self._client = statsd.StatsClient(prefix="kedro")

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        node_timer = self._client.timer(node.name)
        node_timer.start()
        self._timers[node.short_name] = node_timer

    @hook_impl
    def after_node_run(self, node: Node, inputs: Dict[str, Any]) -> None:
        self._timers[node.short_name].stop()
        for dataset_name, dataset_value in inputs.items():
            self._client.gauge(dataset_name + "_size", sys.getsizeof(dataset_value))

    @hook_impl
    def after_pipeline_run(self):
        self._client.incr("run")
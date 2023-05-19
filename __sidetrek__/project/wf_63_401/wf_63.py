import sklearn
import os
import typing
from flytekit import task, workflow, Resources
import sidetrek
from project.wf_63_401.main import Hyperparameters
from project.wf_63_401.main import collect_data
from project.wf_63_401.main import split_data
from project.wf_63_401.main import train_model

@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def dataset_test_org_cc_data()->sidetrek.types.dataset.SidetrekDataset:
	return sidetrek.dataset.build_dataset(io="upload",source="s3://sidetrek-datasets/test-org/cc-data")



_wf_outputs=typing.NamedTuple("WfOutputs",train_model_0=sklearn.ensemble._forest.RandomForestClassifier)
@workflow
def wf_63(_wf_args:Hyperparameters)->_wf_outputs:
	dataset_test_org_cc_data_o0_=dataset_test_org_cc_data()
	collect_data_o0_,collect_data_o1_=collect_data(ds=dataset_test_org_cc_data_o0_)
	split_data_o0_,split_data_o1_,split_data_o2_,split_data_o3_=split_data(hp=_wf_args,X=collect_data_o0_,y=collect_data_o1_)
	train_model_o0_=train_model(X_train=split_data_o0_,y_train=split_data_o2_)
	return _wf_outputs(train_model_o0_)
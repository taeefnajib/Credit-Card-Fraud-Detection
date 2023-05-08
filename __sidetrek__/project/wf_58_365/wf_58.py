import sklearn
import os
import typing
from flytekit import task, workflow, Resources
import sidetrek
from project.wf_58_365.main import Hyperparameters
from project.wf_58_365.main import run_wf
from project.wf_58_365.main import load_data

@task
def dataset_tylo_credit_card_fraud()->sidetrek.types.dataset.SidetrekDataset:
	return sidetrek.dataset.build_dataset(io="upload",source="s3://sidetrek-datasets/tylo/credit-card-fraud")



_wf_outputs=typing.NamedTuple("WfOutputs",run_wf_0=sklearn.ensemble._forest.RandomForestClassifier)
@workflow
def wf_58(_wf_args:Hyperparameters)->_wf_outputs:
	dataset_tylo_credit_card_fraud_o0_=dataset_tylo_credit_card_fraud()
	load_data_o0_,load_data_o1_=load_data(ds=dataset_tylo_credit_card_fraud_o0_)
	run_wf_o0_=run_wf(hp=_wf_args,X=load_data_o0_,y=load_data_o1_)
	return _wf_outputs(run_wf_o0_)
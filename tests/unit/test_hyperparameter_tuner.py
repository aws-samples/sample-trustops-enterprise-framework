"""
Unit tests for the automated hyperparameter tuner.

Tests cover search space configuration generation, Bedrock and SageMaker
trial launching, best trial selection, and default search space.

Requirements: 4.6
"""

from unittest.mock import MagicMock

from src.data_models.fine_tuning import HyperparameterConfig
from src.fine_tuning.hyperparameter_tuner import (
    HyperparameterRange,
    HyperparameterSearchSpace,
    TuningTrialResult,
    TuningResult,
    default_search_space,
    generate_configurations,
    launch_tuning_trials_bedrock,
    launch_tuning_trials_sagemaker,
    select_best_trial,
)


# --- Helpers ---


def _base_config() -> HyperparameterConfig:
    return HyperparameterConfig(
        learning_rate=1e-5,
        epochs=3,
        batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
    )


def _mock_bedrock_client(job_arn_prefix="arn:aws:bedrock:us-east-1:123:job/"):
    client = MagicMock()
    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        return {"jobArn": f"{job_arn_prefix}{call_count['n']}"}

    client.create_model_customization_job.side_effect = side_effect
    return client


def _mock_sagemaker_client(arn_prefix="arn:aws:sagemaker:us-east-1:123:job/"):
    client = MagicMock()
    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        return {"TrainingJobArn": f"{arn_prefix}{call_count['n']}"}

    client.create_training_job.side_effect = side_effect
    return client


# --- generate_configurations ---


class TestGenerateConfigurations:
    def test_empty_search_space_returns_base(self):
        base = _base_config()
        space = HyperparameterSearchSpace()
        configs = generate_configurations(base, space)
        assert len(configs) == 1
        assert configs[0].learning_rate == base.learning_rate

    def test_single_param_range(self):
        base = _base_config()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5, 5e-6]),
        )
        configs = generate_configurations(base, space)
        assert len(configs) == 2
        rates = {c.learning_rate for c in configs}
        assert rates == {1e-5, 5e-6}
        # Other params stay at base values
        for c in configs:
            assert c.epochs == base.epochs
            assert c.batch_size == base.batch_size

    def test_two_param_cartesian_product(self):
        base = _base_config()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5, 5e-6]),
            epochs=HyperparameterRange(values=[2, 3, 4]),
        )
        configs = generate_configurations(base, space)
        assert len(configs) == 6  # 2 * 3

    def test_three_param_cartesian_product(self):
        base = _base_config()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5, 5e-6]),
            epochs=HyperparameterRange(values=[2, 3]),
            batch_size=HyperparameterRange(values=[4, 8]),
        )
        configs = generate_configurations(base, space)
        assert len(configs) == 8  # 2 * 2 * 2

    def test_preserves_non_searched_params(self):
        base = HyperparameterConfig(
            learning_rate=1e-5,
            epochs=3,
            batch_size=8,
            warmup_steps=200,
            weight_decay=0.02,
            max_seq_length=4096,
            lora_rank=16,
        )
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5]),
        )
        configs = generate_configurations(base, space)
        assert len(configs) == 1
        assert configs[0].warmup_steps == 200
        assert configs[0].weight_decay == 0.02
        assert configs[0].max_seq_length == 4096
        assert configs[0].lora_rank == 16

    def test_empty_values_list_returns_base(self):
        base = _base_config()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[]),
        )
        configs = generate_configurations(base, space)
        assert len(configs) == 1


# --- launch_tuning_trials_bedrock ---


class TestLaunchTuningTrialsBedrock:
    def test_launches_correct_number_of_jobs(self):
        client = _mock_bedrock_client()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5, 5e-6]),
        )
        trials = launch_tuning_trials_bedrock(
            bedrock_client=client,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        assert len(trials) == 2
        assert client.create_model_customization_job.call_count == 2

    def test_trial_ids_are_unique(self):
        client = _mock_bedrock_client()
        space = HyperparameterSearchSpace(
            epochs=HyperparameterRange(values=[2, 3, 4]),
        )
        trials = launch_tuning_trials_bedrock(
            bedrock_client=client,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        ids = [t.trial_id for t in trials]
        assert len(set(ids)) == 3

    def test_successful_trials_have_arns(self):
        client = _mock_bedrock_client()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5]),
        )
        trials = launch_tuning_trials_bedrock(
            bedrock_client=client,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        assert trials[0].success is True
        assert trials[0].job_arn != ""

    def test_failed_trial_on_api_error(self):
        client = MagicMock()
        client.create_model_customization_job.side_effect = Exception("boom")
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5]),
        )
        trials = launch_tuning_trials_bedrock(
            bedrock_client=client,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        assert trials[0].success is False
        assert trials[0].error != ""

    def test_validation_data_passed_through(self):
        client = _mock_bedrock_client()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5]),
        )
        launch_tuning_trials_bedrock(
            bedrock_client=client,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
            validation_data_s3_uri="s3://bucket/val.jsonl",
        )
        call_kwargs = client.create_model_customization_job.call_args[1]
        assert "validationDataConfig" in call_kwargs


# --- launch_tuning_trials_sagemaker ---


class TestLaunchTuningTrialsSageMaker:
    def test_launches_correct_number_of_jobs(self):
        client = _mock_sagemaker_client()
        space = HyperparameterSearchSpace(
            batch_size=HyperparameterRange(values=[4, 8, 16]),
        )
        image = "123.dkr.ecr.us-east-1.amazonaws.com/t:1"
        trials = launch_tuning_trials_sagemaker(
            sagemaker_client=client,
            container_image_uri=image,
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        assert len(trials) == 3
        assert client.create_training_job.call_count == 3

    def test_successful_trials(self):
        client = _mock_sagemaker_client()
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5]),
        )
        image = "123.dkr.ecr.us-east-1.amazonaws.com/t:1"
        trials = launch_tuning_trials_sagemaker(
            sagemaker_client=client,
            container_image_uri=image,
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        assert trials[0].success is True
        assert trials[0].job_arn != ""

    def test_failed_trial_on_api_error(self):
        client = MagicMock()
        client.create_training_job.side_effect = Exception("denied")
        space = HyperparameterSearchSpace(
            learning_rate=HyperparameterRange(values=[1e-5]),
        )
        image = "123.dkr.ecr.us-east-1.amazonaws.com/t:1"
        trials = launch_tuning_trials_sagemaker(
            sagemaker_client=client,
            container_image_uri=image,
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123:role/Role",
            base_config=_base_config(),
            search_space=space,
        )
        assert trials[0].success is False
        assert "denied" in trials[0].error


# --- select_best_trial ---


class TestSelectBestTrial:
    def test_selects_lowest_validation_loss(self):
        trials = [
            TuningTrialResult(
                trial_id="t-0",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.5,
            ),
            TuningTrialResult(
                trial_id="t-1",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.3,
            ),
            TuningTrialResult(
                trial_id="t-2",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.7,
            ),
        ]
        result = select_best_trial(trials)
        assert result.best_trial is not None
        assert result.best_trial.trial_id == "t-1"
        assert result.best_validation_loss == 0.3

    def test_ignores_failed_trials(self):
        trials = [
            TuningTrialResult(
                trial_id="t-0",
                hyperparameters=_base_config(),
                success=False,
                validation_loss=0.1,
            ),
            TuningTrialResult(
                trial_id="t-1",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.5,
            ),
        ]
        result = select_best_trial(trials)
        assert result.best_trial is not None
        assert result.best_trial.trial_id == "t-1"
        assert result.completed_trials == 1

    def test_ignores_trials_without_validation_loss(self):
        trials = [
            TuningTrialResult(
                trial_id="t-0",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=None,
            ),
            TuningTrialResult(
                trial_id="t-1",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.4,
            ),
        ]
        result = select_best_trial(trials)
        assert result.best_trial is not None
        assert result.best_trial.trial_id == "t-1"
        assert result.completed_trials == 1

    def test_no_completed_trials(self):
        trials = [
            TuningTrialResult(
                trial_id="t-0",
                hyperparameters=_base_config(),
                success=False,
                error="failed",
            ),
        ]
        result = select_best_trial(trials)
        assert result.best_trial is None
        assert result.best_validation_loss is None
        assert result.completed_trials == 0

    def test_empty_trials_list(self):
        result = select_best_trial([])
        assert result.best_trial is None
        assert result.total_trials == 0
        assert result.completed_trials == 0

    def test_total_and_completed_counts(self):
        trials = [
            TuningTrialResult(
                trial_id="t-0",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.5,
            ),
            TuningTrialResult(
                trial_id="t-1",
                hyperparameters=_base_config(),
                success=False,
                error="err",
            ),
            TuningTrialResult(
                trial_id="t-2",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=0.3,
            ),
        ]
        result = select_best_trial(trials)
        assert result.total_trials == 3
        assert result.completed_trials == 2

    def test_all_trials_stored(self):
        trials = [
            TuningTrialResult(
                trial_id=f"t-{i}",
                hyperparameters=_base_config(),
                success=True,
                validation_loss=float(i),
            )
            for i in range(5)
        ]
        result = select_best_trial(trials)
        assert len(result.trials) == 5


# --- default_search_space ---


class TestDefaultSearchSpace:
    def test_has_learning_rate_range(self):
        space = default_search_space()
        assert space.learning_rate is not None
        assert len(space.learning_rate.values) > 0

    def test_has_epochs_range(self):
        space = default_search_space()
        assert space.epochs is not None
        assert len(space.epochs.values) > 0

    def test_has_batch_size_range(self):
        space = default_search_space()
        assert space.batch_size is not None
        assert len(space.batch_size.values) > 0

    def test_generates_multiple_configs(self):
        space = default_search_space()
        configs = generate_configurations(_base_config(), space)
        assert len(configs) > 1


# --- TuningTrialResult defaults ---


class TestTuningTrialResultDefaults:
    def test_defaults(self):
        trial = TuningTrialResult(
            trial_id="t-0",
            hyperparameters=_base_config(),
        )
        assert trial.success is True
        assert trial.job_arn == ""
        assert trial.validation_loss is None
        assert trial.error == ""


# --- TuningResult defaults ---


class TestTuningResultDefaults:
    def test_defaults(self):
        result = TuningResult()
        assert result.best_trial is None
        assert result.trials == []
        assert result.total_trials == 0
        assert result.completed_trials == 0
        assert result.best_validation_loss is None

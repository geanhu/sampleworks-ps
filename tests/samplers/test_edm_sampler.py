"""Tests for the EDM sampler's schedule and configuration."""
import pytest
from tests.mocks import (
    MockFlowModelWrapper
)

import torch
from sampleworks.core.samplers.edm import (
    EDMSamplerConfig,
    AF3EDMSampler
)

class TestAsynchDiff:
    """Test asynchronous diffusion in the EDM sampler."""

    def test_flexibility_prior_tensor(self):
        """Test that flexibility_prior is tensor runtime"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sampler_config = EDMSamplerConfig(
            device = device,
            flexibility_prior=[0, 0]
        )
        sampler = AF3EDMSampler(
            config = sampler_config
        )

        assert torch.is_tensor(sampler.flexibility_prior), (
            "flexibility_prior is not tensor at runtime"
        )
        assert sampler.flexibility_prior.device.type == device, (
            "flexibility_prior is not placed on correct device"
        )

    # ---- step scale
    @pytest.mark.parametrize(
        "bound",
        [0.8, 2.0],
        ids=["low", "high"],
    )
    def test_invalid_step_scale_bound(self, bound: float):
        """Test that invalid step scale bounds throw error."""
        with pytest.raises(ValueError):
            EDMSamplerConfig(
                asynchronous_diffusion_parameter=['step_scale'],
                asynchronous_diffusion_bound=bound
            )

    @pytest.mark.parametrize(
        "parameter",
        [None, ['noise_scale']],
        ids=["none_selected", "other_selected"],
    )
    def test_not_applied_if_not_selected(self, parameter: list | None):
        """Test that position-wise step scale is not used if not
        selected in asynchronous_diffusion_parameter.
        """
        sampler_config = EDMSamplerConfig(
            # turn augmentation off for deterministic output
            augmentation=False,
            # prior of 1 (highest value) and bound of 1
            # should set the step_scale to be 1
            flexibility_prior=[1],
            asynchronous_diffusion_parameter=parameter,
            asynchronous_diffusion_bound=1
        )
        sampler = AF3EDMSampler(
            config = sampler_config
        )

        mock_state = torch.tensor(
            [[1, 1, 1]],
            dtype=torch.float32
        )
        mock_wrapper = MockFlowModelWrapper(
            num_atoms=1,
            target=mock_state
        )
        mock_StepParams = sampler.get_context_for_step(
            # pick step with churn shutoff so noise
            # is not re-added and output is 
            # deterministic
            step_index=199,
            schedule=sampler.compute_schedule(
                num_steps=200
            )
        )
        output = sampler.step(
            state = mock_state,
            # create mock ModelWrapper that returns same state
            model_wrapper=mock_wrapper,
            context = mock_StepParams,
            features=mock_wrapper.featurize(
                structure={}
            )
        )
        torch.testing.assert_close(
            actual = output.state,
            expected = torch.tensor(
                [[0.2515, 0.2515, 0.2515]],
                dtype=torch.float32
            ),
            rtol=0,
            atol=0.01,
            msg=lambda msg: (
                "Output state seems to have modified step scale despite step_scale "
                "not being provided to asynchronous_diffusion_parameter."
                f"\n{msg}"
            )
        )
        

    def test_not_applied_upon_shutoff(self):
        """Test that position-wise step scale is not used after churn shutoff,
        if enabled.
        """
        sampler_config = EDMSamplerConfig(
            # turn augmentation off for deterministic output
            augmentation=False,
            # prior of 1 (highest value) and bound of 1
            # should set the step_scale to be 1
            flexibility_prior=[1],
            asynchronous_diffusion_parameter=['step_scale'],
            asynchronous_diffusion_bound=1,
            # shutoff enabled
            asynchronous_diffusion_shutoff=True
        )
        sampler = AF3EDMSampler(
            config = sampler_config
        )

        mock_state = torch.tensor(
            [[1, 1, 1]],
            dtype=torch.float32
        )
        mock_wrapper = MockFlowModelWrapper(
            num_atoms=1,
            target=mock_state
        )
        mock_StepParams = sampler.get_context_for_step(
            step_index=199,
            schedule=sampler.compute_schedule(
                num_steps=200
            )
        )
        output = sampler.step(
            state = mock_state,
            # create mock ModelWrapper that returns same state
            model_wrapper=mock_wrapper,
            context = mock_StepParams,
            features=mock_wrapper.featurize(
                structure={}
            )
        )
        torch.testing.assert_close(
            actual = output.state,
            expected = torch.tensor(
                [[0.2515, 0.2515, 0.2515]],
                dtype=torch.float32
            ),
            rtol=0,
            atol=0.01,
            msg=lambda msg: (
                "Output state seems to have modified step scale, but "
                "asynchronous diffusion is supposed to be shut off "
                "after churn is disabled. "
                f"\n{msg}"
            )
        )
    
    def test_step_scale(self):
        """Test whether position-wise noise scale is correctly applied.
        """
        sampler_config = EDMSamplerConfig(
            # turn augmentation off for deterministic output
            augmentation=False,
            # prior of 1 (highest value) and bound of 1
            # should set the step_scale to be 1
            flexibility_prior=[1, 0],
            asynchronous_diffusion_parameter=['step_scale'],
            asynchronous_diffusion_bound=1
        )
        sampler = AF3EDMSampler(
            config = sampler_config
        )

        mock_state = torch.tensor(
            [[1, 1, 1],
             [1, 1, 1]],
            dtype=torch.float32
        )
        mock_wrapper = MockFlowModelWrapper(
            num_atoms=2,
            target=mock_state
        )
        mock_StepParams = sampler.get_context_for_step(
            # pick step with churn shutoff so noise
            # is not re-added and output is 
            # deterministic
            step_index=199,
            schedule=sampler.compute_schedule(
                num_steps=200
            )
        )
        output = sampler.step(
            state = mock_state,
            # create mock ModelWrapper that returns same state
            model_wrapper=mock_wrapper,
            context = mock_StepParams,
            features=mock_wrapper.featurize(
                structure={}
            )
        )
        torch.testing.assert_close(
            actual = output.state,
            expected = torch.tensor(
                [[0.169, 0.169, 0.169],
                 [0.2515, 0.2515, 0.2515]],
                dtype=torch.float32
            ),
            rtol=0,
            atol=0.01,
            msg=lambda msg: (
                "Output state seems to have modified step scale despite step_scale "
                "not being provided to asynchronous_diffusion_parameter."
                f"\n{msg}"
            )
        )
    
    # ---- noise scale

    # ---- guidance weight

    
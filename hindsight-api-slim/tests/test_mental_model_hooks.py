"""Unit tests for mental model operation validator hooks.

Tests that the operation validator hooks are called correctly for
mental model GET and refresh operations.
"""

import pytest

from hindsight_api.extensions.operation_validator import (
    MentalModelGetContext,
    MentalModelGetResult,
    MentalModelRefreshResult,
    OperationValidatorExtension,
    ValidationResult,
)


class TestMentalModelGetContextDataclass:
    """Tests for MentalModelGetContext dataclass."""

    def test_create_context(self):
        """Test creating a MentalModelGetContext."""
        from unittest.mock import MagicMock

        request_context = MagicMock()
        ctx = MentalModelGetContext(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=request_context,
        )

        assert ctx.bank_id == "bank-1"
        assert ctx.mental_model_id == "mm-1"
        assert ctx.request_context is request_context


class TestMentalModelGetResultDataclass:
    """Tests for MentalModelGetResult dataclass."""

    def test_create_result_success(self):
        """Test creating a successful MentalModelGetResult."""
        from unittest.mock import MagicMock

        request_context = MagicMock()
        result = MentalModelGetResult(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=request_context,
            output_tokens=250,
        )

        assert result.bank_id == "bank-1"
        assert result.mental_model_id == "mm-1"
        assert result.output_tokens == 250
        assert result.success is True
        assert result.error is None

    def test_create_result_failure(self):
        """Test creating a failed MentalModelGetResult."""
        from unittest.mock import MagicMock

        result = MentalModelGetResult(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=MagicMock(),
            output_tokens=0,
            success=False,
            error="Not found",
        )

        assert result.success is False
        assert result.error == "Not found"


class TestMentalModelRefreshResultDataclass:
    """Tests for MentalModelRefreshResult dataclass."""

    def test_create_result_with_all_fields(self):
        """Test creating a MentalModelRefreshResult with all fields."""
        from unittest.mock import MagicMock

        result = MentalModelRefreshResult(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=MagicMock(),
            query_tokens=50,
            output_tokens=500,
            context_tokens=0,
            facts_used=10,
            mental_models_used=2,
        )

        assert result.query_tokens == 50
        assert result.output_tokens == 500
        assert result.context_tokens == 0
        assert result.facts_used == 10
        assert result.mental_models_used == 2
        assert result.success is True
        assert result.error is None

    def test_create_result_failure(self):
        """Test creating a failed MentalModelRefreshResult."""
        from unittest.mock import MagicMock

        result = MentalModelRefreshResult(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=MagicMock(),
            query_tokens=50,
            output_tokens=0,
            context_tokens=0,
            facts_used=0,
            mental_models_used=0,
            success=False,
            error="Reflect failed",
        )

        assert result.success is False
        assert result.error == "Reflect failed"


class TestDefaultHookBehavior:
    """Tests for default (no-op) behavior of mental model hooks on OperationValidatorExtension."""

    @pytest.fixture
    def validator(self):
        """Create a concrete subclass for testing default behavior."""
        from unittest.mock import MagicMock

        # Create a concrete subclass that implements the abstract methods
        class TestValidator(OperationValidatorExtension):
            async def validate_retain(self, ctx):
                return ValidationResult.accept()

            async def validate_recall(self, ctx):
                return ValidationResult.accept()

            async def validate_reflect(self, ctx):
                return ValidationResult.accept()

        return TestValidator(config={})

    @pytest.mark.asyncio
    async def test_validate_mental_model_get_default_accepts(self, validator):
        """Test that default validate_mental_model_get accepts."""
        from unittest.mock import MagicMock

        ctx = MentalModelGetContext(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=MagicMock(),
        )

        result = await validator.validate_mental_model_get(ctx)

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_on_mental_model_get_complete_default_noop(self, validator):
        """Test that default on_mental_model_get_complete is a no-op."""
        from unittest.mock import MagicMock

        result = MentalModelGetResult(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=MagicMock(),
            output_tokens=100,
        )

        # Should not raise
        await validator.on_mental_model_get_complete(result)

    @pytest.mark.asyncio
    async def test_on_mental_model_refresh_complete_default_noop(self, validator):
        """Test that default on_mental_model_refresh_complete is a no-op."""
        from unittest.mock import MagicMock

        result = MentalModelRefreshResult(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=MagicMock(),
            query_tokens=50,
            output_tokens=500,
            context_tokens=0,
            facts_used=5,
            mental_models_used=1,
        )

        # Should not raise
        await validator.on_mental_model_refresh_complete(result)


class TestExportsAvailable:
    """Test that mental model hooks are properly exported."""

    def test_imports_from_extensions_package(self):
        """Test that all mental model types can be imported from hindsight_api.extensions."""
        from hindsight_api.extensions import (
            MentalModelGetContext,
            MentalModelGetResult,
            MentalModelRefreshResult,
        )

        assert MentalModelGetContext is not None
        assert MentalModelGetResult is not None
        assert MentalModelRefreshResult is not None


class TestMentalModelHistoryOperationValidation:
    """Tests that get_mental_model_history invokes operation validator hooks."""

    @pytest.mark.asyncio
    async def test_get_mental_model_history_invokes_validator(self):
        """Test get_mental_model_history calls validate_mental_model_get and rejects when disallowed."""
        from unittest.mock import AsyncMock, MagicMock
        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.extensions.operation_validator import (
            OperationValidationError,
            OperationValidatorExtension,
            ValidationResult,
        )

        class RejectingValidator(OperationValidatorExtension):
            async def validate_retain(self, ctx):
                return ValidationResult.accept()

            async def validate_recall(self, ctx):
                return ValidationResult.accept()

            async def validate_reflect(self, ctx):
                return ValidationResult.accept()

            async def validate_mental_model_get(self, ctx):
                return ValidationResult.reject("caller is not authorized for the requested bank")

        validator = RejectingValidator(config={})
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._operation_validator = validator
        engine._authenticate_tenant = AsyncMock()
        engine._consume_preauthorized_mental_model_operation = MagicMock(return_value=False)

        request_context = MagicMock()

        with pytest.raises(OperationValidationError) as exc_info:
            await engine.get_mental_model_history("bank-1", "mm-1", request_context=request_context)

        assert "caller is not authorized" in str(exc_info.value)
        engine._authenticate_tenant.assert_awaited_once_with(request_context)

    @pytest.mark.asyncio
    async def test_get_mental_model_history_completes_post_hook_with_structured_reflect_response(self):
        """Test get_mental_model_history handles structured objects in previous_reflect_response and computes output_tokens accurately."""
        import json
        from unittest.mock import AsyncMock, MagicMock
        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.extensions.operation_validator import (
            OperationValidatorExtension,
            ValidationResult,
        )

        mock_get_complete = AsyncMock()

        class AcceptingValidator(OperationValidatorExtension):
            async def validate_retain(self, ctx):
                return ValidationResult.accept()

            async def validate_recall(self, ctx):
                return ValidationResult.accept()

            async def validate_reflect(self, ctx):
                return ValidationResult.accept()

            async def validate_mental_model_get(self, ctx):
                return ValidationResult.accept()

            async def on_mental_model_get_complete(self, result):
                await mock_get_complete(result)

        validator = AcceptingValidator(config={})
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._operation_validator = validator
        engine._authenticate_tenant = AsyncMock()
        engine._consume_preauthorized_mental_model_operation = MagicMock(return_value=False)

        previous_content = "User prefers concise Python 3.12 code."
        previous_reflect_response = {"answer": "Detailed analysis", "confidence": 0.95, "tags": ["python", "style"]}
        reflect_json_str = json.dumps(previous_reflect_response)

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": "mm-1"}
        mock_conn.fetch.return_value = [
            {
                "content": json.dumps({
                    "previous_content": previous_content,
                    "previous_reflect_response": previous_reflect_response,
                }),
                "changed_at": "2026-08-27T00:00:00",
            }
        ]

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None

        engine._get_backend = AsyncMock()
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hindsight_api.engine.memory_engine.acquire_with_retry", lambda backend: mock_cm)

            request_context = MagicMock()
            res = await engine.get_mental_model_history("bank-1", "mm-1", request_context=request_context)

            assert res == [
                {
                    "previous_content": previous_content,
                    "previous_reflect_response": previous_reflect_response,
                    "changed_at": "2026-08-27T00:00:00",
                }
            ]
            mock_get_complete.assert_awaited_once()
            call_arg = mock_get_complete.call_args[0][0]
            assert call_arg.bank_id == "bank-1"
            assert call_arg.mental_model_id == "mm-1"
            assert call_arg.success is True

            # Calculate expected token measurement
            expected_total_len = len(previous_content) + len(reflect_json_str)
            expected_tokens = expected_total_len // 4
            assert call_arg.output_tokens == expected_tokens

    @pytest.mark.asyncio
    async def test_get_mental_model_history_preauthorized_skips_validation_but_runs_hook(self):
        """Test preauthorized calls skip validate_mental_model_get but still record completion hook."""
        from unittest.mock import AsyncMock, MagicMock
        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.extensions.operation_validator import (
            OperationValidatorExtension,
            ValidationResult,
        )

        mock_validate_get = AsyncMock(return_value=ValidationResult.reject("Should not be called"))
        mock_get_complete = AsyncMock()

        class PreauthorizedValidator(OperationValidatorExtension):
            async def validate_retain(self, ctx):
                return ValidationResult.accept()

            async def validate_recall(self, ctx):
                return ValidationResult.accept()

            async def validate_reflect(self, ctx):
                return ValidationResult.accept()

            async def validate_mental_model_get(self, ctx):
                return await mock_validate_get(ctx)

            async def on_mental_model_get_complete(self, result):
                await mock_get_complete(result)

        validator = PreauthorizedValidator(config={})
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._operation_validator = validator
        engine._authenticate_tenant = AsyncMock()
        # Preauthorized operation returns True
        engine._consume_preauthorized_mental_model_operation = MagicMock(return_value=True)

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": "mm-1"}
        mock_conn.fetch.return_value = [
            {"content": '{"previous_content": "preauth item"}', "changed_at": "2026-08-27T00:00:00"}
        ]

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None

        engine._get_backend = AsyncMock()
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hindsight_api.engine.memory_engine.acquire_with_retry", lambda backend: mock_cm)

            request_context = MagicMock()
            res = await engine.get_mental_model_history("bank-1", "mm-1", request_context=request_context)

            assert res is not None
            mock_validate_get.assert_not_called()
            mock_get_complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_mental_model_history_hook_error_is_non_fatal(self):
        """Test that errors in accounting/completion hook do not raise 500 or break response."""
        from unittest.mock import AsyncMock, MagicMock
        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.extensions.operation_validator import (
            OperationValidatorExtension,
            ValidationResult,
        )

        class FaultyHookValidator(OperationValidatorExtension):
            async def validate_retain(self, ctx):
                return ValidationResult.accept()

            async def validate_recall(self, ctx):
                return ValidationResult.accept()

            async def validate_reflect(self, ctx):
                return ValidationResult.accept()

            async def validate_mental_model_get(self, ctx):
                return ValidationResult.accept()

            async def on_mental_model_get_complete(self, result):
                raise RuntimeError("Hook database connection failed")

        validator = FaultyHookValidator(config={})
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._operation_validator = validator
        engine._authenticate_tenant = AsyncMock()
        engine._consume_preauthorized_mental_model_operation = MagicMock(return_value=False)

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": "mm-1"}
        mock_conn.fetch.return_value = [
            {"content": '{"previous_content": "data"}', "changed_at": "2026-08-27T00:00:00"}
        ]

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None

        engine._get_backend = AsyncMock()
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hindsight_api.engine.memory_engine.acquire_with_retry", lambda backend: mock_cm)

            request_context = MagicMock()
            # Should NOT raise RuntimeError or 500
            res = await engine.get_mental_model_history("bank-1", "mm-1", request_context=request_context)

            assert res == [
                {
                    "previous_content": "data",
                    "previous_reflect_response": None,
                    "changed_at": "2026-08-27T00:00:00",
                }
            ]



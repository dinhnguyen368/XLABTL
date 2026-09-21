import json
import math
import numbers
from pathlib import Path
from typing import Any, Dict

from src.utils import setup_logger


logger = setup_logger("AdaptiveConfig")


class NoiseAdaptiveConfig:
    """
    Load and apply frozen noise-proxy-to-Bilateral parameter rules from E2.

    The adaptive signal is 'noise_measure', which is an image-level
    high-frequency residual noise proxy measured from a homogeneous ROI.
    """

    REQUIRED_TOP_LEVEL = {
        "thresholds",
        "low_noise",
        "medium_noise",
        "high_noise",
    }

    REQUIRED_RULE_FIELDS = {
        "d",
        "sigma_color",
        "sigma_space",
        "dev_strict_f1",
    }

    def __init__(self, rules_path: str | Path):
        self.rules_path = Path(rules_path)

        if not self.rules_path.exists():
            logger.critical(
                f"Adaptive rules not found at {self.rules_path}. "
                f"Run E2 first to freeze rules."
            )
            raise SystemExit(1)

        try:
            with self.rules_path.open("r", encoding="utf-8") as f:
                self.rules = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.critical(f"Failed to read adaptive rules: {exc}")
            raise SystemExit(1)

        self._validate_schema()

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, numbers.Number) and math.isfinite(float(value))

    def _validate_schema(self) -> None:
        if set(self.rules.keys()) != self.REQUIRED_TOP_LEVEL:
            logger.critical(
                f"Exact schema failed. Expected exactly "
                f"{self.REQUIRED_TOP_LEVEL}; "
                f"found {set(self.rules.keys())}."
            )
            raise SystemExit(1)

        thresholds = self.rules["thresholds"]

        if (
            not isinstance(thresholds, dict)
            or set(thresholds.keys()) != {"t1", "t2"}
        ):
            logger.critical(
                "Threshold schema failed. Expected exactly t1 and t2."
            )
            raise SystemExit(1)

        t1 = thresholds["t1"]
        t2 = thresholds["t2"]

        if not self._is_finite_number(t1) or not self._is_finite_number(t2):
            logger.critical(
                "t1 and t2 must be finite numeric values."
            )
            raise SystemExit(1)

        t1 = float(t1)
        t2 = float(t2)

        if t1 <= 0 or t2 <= 0 or t1 >= t2:
            logger.critical(
                f"Thresholds must satisfy 0 < t1 < t2. "
                f"Got t1={t1}, t2={t2}."
            )
            raise SystemExit(1)

        for bin_name in (
            "low_noise",
            "medium_noise",
            "high_noise",
        ):
            rule = self.rules[bin_name]

            if not isinstance(rule, dict):
                logger.critical(
                    f"Rule {bin_name} must be an object."
                )
                raise SystemExit(1)

            if set(rule.keys()) != self.REQUIRED_RULE_FIELDS:
                logger.critical(
                    f"Rule {bin_name} must contain exactly "
                    f"{self.REQUIRED_RULE_FIELDS}; "
                    f"found {set(rule.keys())}."
                )
                raise SystemExit(1)

            for param in self.REQUIRED_RULE_FIELDS:
                if not self._is_finite_number(rule[param]):
                    logger.critical(
                        f"{bin_name}.{param} must be finite and numeric."
                    )
                    raise SystemExit(1)

            if (
                float(rule["d"]) <= 0
                or float(rule["d"]) != int(rule["d"])
            ):
                logger.critical(
                    f"{bin_name}.d must be a positive integer."
                )
                raise SystemExit(1)

            if (
                float(rule["sigma_color"]) < 0
                or float(rule["sigma_space"]) < 0
            ):
                logger.critical(
                    f"{bin_name}.sigma_color and sigma_space "
                    f"must be >= 0."
                )
                raise SystemExit(1)

            if not 0.0 <= float(rule["dev_strict_f1"]) <= 1.0:
                logger.critical(
                    f"{bin_name}.dev_strict_f1 must be in [0, 1]."
                )
                raise SystemExit(1)

    def get_config(
        self,
        noise_value: float,
        default_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map measured noise proxy to a frozen Bilateral configuration.
        """

        if (
            not self._is_finite_number(noise_value)
            or float(noise_value) < 0
        ):
            raise ValueError(
                f"Invalid noise_measure value: {noise_value}"
            )

        cfg = default_config.copy()

        t1 = float(self.rules["thresholds"]["t1"])
        t2 = float(self.rules["thresholds"]["t2"])

        value = float(noise_value)

        if value <= t1:
            rule = self.rules["low_noise"]
        elif value <= t2:
            rule = self.rules["medium_noise"]
        else:
            rule = self.rules["high_noise"]

        cfg["bilateral_d"] = int(rule["d"])
        cfg["bilateral_sigma_color"] = float(
            rule["sigma_color"]
        )
        cfg["bilateral_sigma_space"] = float(
            rule["sigma_space"]
        )

        return cfg

    def get_heavy_config(
        self,
        default_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return the frozen high-noise configuration.
        Used by E4 as the fixed stress-test configuration.
        """

        cfg = default_config.copy()
        rule = self.rules["high_noise"]

        cfg["bilateral_d"] = int(rule["d"])
        cfg["bilateral_sigma_color"] = float(
            rule["sigma_color"]
        )
        cfg["bilateral_sigma_space"] = float(
            rule["sigma_space"]
        )

        return cfg


# Backward-compatible alias in case another local module still imports
# the old class name.
ISOAdaptiveConfig = NoiseAdaptiveConfig
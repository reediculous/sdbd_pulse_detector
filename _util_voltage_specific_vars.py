"""Voltage-specific parameters for event detection pipeline."""

VOLTAGE_SHIFT = {
    1600: 0.0000525,
    1800: 0.0000525,
    2000: 0.0000525,
    2200: -0.00003,
    2400: 0.00001
}

CAPACITOR_CURRENT_AMP = {
    1600: 1.068774e-04,
    1800: 1.068774e-04,
    2000: 1.188774e-04,
    2200: 1.30627933e-04,
    2400: 1.603161e-04
}

CAPACITOR_CURRENT_DELAY = -7.068553539992742e-06


def get_voltage_shift(voltage, default=None):
    """Get voltage shift for a given voltage."""
    if default is None:
        if voltage not in VOLTAGE_SHIFT or VOLTAGE_SHIFT[voltage] is None:
            raise KeyError(f"Voltage shift not defined for voltage {voltage}. "
                         f"Available voltages: {[k for k, v in VOLTAGE_SHIFT.items() if v is not None]}")
        return VOLTAGE_SHIFT[voltage]
    return VOLTAGE_SHIFT.get(voltage, default)


def get_capacitor_current_amp(voltage, default=None):
    """Get capacitor current amplitude for a given voltage."""
    if default is None:
        if voltage not in CAPACITOR_CURRENT_AMP or CAPACITOR_CURRENT_AMP[voltage] is None:
            raise KeyError(f"Capacitor current amplitude not defined for voltage {voltage}. "
                          f"Available voltages: {[k for k, v in CAPACITOR_CURRENT_AMP.items() if v is not None]}")
        return CAPACITOR_CURRENT_AMP[voltage]
    return CAPACITOR_CURRENT_AMP.get(voltage, default)

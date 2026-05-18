import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

csv_file = r""    # Your exported UXA IQ CSV file
fs = 20e6                      # IQ sample rate from UXA, change this
symbol_rate = 1e6              # Bluetooth EDR2M symbol rate
prbs9_seed = 0x1FF             # PRBS9 seed = 1FF

# Bluetooth EDR DEVM limits
LIMIT_RMS = 20.0               # %
LIMIT_PEAK = 35.0              # %
LIMIT_99 = 30.0                # %

# If your CSV has no header and only two columns, keep this True
AUTO_FIND_NUMERIC_COLUMNS = False


# ============================================================
# LOAD IQ
# ============================================================

def load_iq_csv(filename):
    df = pd.read_csv(filename)

    if AUTO_FIND_NUMERIC_COLUMNS:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            raise ValueError("CSV must contain at least two numeric columns: I and Q.")
        i = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        q = numeric_df.iloc[:, 1].to_numpy(dtype=float)
    else:
        i = df["I"].to_numpy(dtype=float)
        q = df["Q"].to_numpy(dtype=float)

    x = i + 1j * q

    # Remove DC
    x = x - np.mean(x)

    # Normalize RMS power
    pwr = np.mean(np.abs(x) ** 2)
    if pwr <= 0:
        raise ValueError("IQ signal has zero power.")

    x = x / np.sqrt(pwr)

    return x


# ============================================================
# PRBS9 GENERATOR
# ============================================================

def generate_prbs9_bits(num_bits, seed=0x1FF):
    """
    PRBS9 polynomial:
        x^9 + x^5 + 1

    Seed:
        0x1FF = 111111111b

    LSB-output implementation.
    If instrument bit order is different, brute-force options later help.
    """

    if seed <= 0 or seed > 0x1FF:
        raise ValueError("PRBS9 seed must be 1 to 0x1FF.")

    reg = seed & 0x1FF
    bits = []

    for _ in range(num_bits):
        out = reg & 0x1
        bits.append(out)

        feedback = ((reg >> 0) ^ (reg >> 4)) & 0x1

        reg = (reg >> 1) | (feedback << 8)
        reg &= 0x1FF

    return np.array(bits, dtype=np.uint8)


# ============================================================
# PI/4-DQPSK MAPPING
# ============================================================

def bits_to_pi4dqpsk_diff_symbols(
    bits,
    mapping_id=0,
    invert_bits=False,
    swap_bit_pair=False
):
    """
    Converts PRBS9 bit pairs into ideal pi/4-DQPSK differential symbols.

    Tries several mappings because real instruments enjoy hiding bit convention.
    """

    bits = np.array(bits, dtype=np.uint8)

    if invert_bits:
        bits = 1 - bits

    if len(bits) % 2 != 0:
        bits = bits[:-1]

    b0 = bits[0::2]
    b1 = bits[1::2]

    if swap_bit_pair:
        b0, b1 = b1, b0

    pair = (b0 << 1) | b1

    mappings = []

    # Mapping 0: common Gray-like pi/4-DQPSK
    mappings.append({
        0b00: np.pi / 4,
        0b01: 3 * np.pi / 4,
        0b11: -3 * np.pi / 4,
        0b10: -np.pi / 4,
    })

    # Mapping 1: sign reversed
    mappings.append({
        0b00: -np.pi / 4,
        0b01: -3 * np.pi / 4,
        0b11: 3 * np.pi / 4,
        0b10: np.pi / 4,
    })

    # Mapping 2: alternate ordering
    mappings.append({
        0b00: np.pi / 4,
        0b01: -np.pi / 4,
        0b11: -3 * np.pi / 4,
        0b10: 3 * np.pi / 4,
    })

    # Mapping 3: alternate ordering sign reversed
    mappings.append({
        0b00: -np.pi / 4,
        0b01: np.pi / 4,
        0b11: 3 * np.pi / 4,
        0b10: -3 * np.pi / 4,
    })

    selected = mappings[mapping_id]

    phases = np.array([selected[int(p)] for p in pair])
    return np.exp(1j * phases)


# ============================================================
# FREQUENCY OFFSET CORRECTION
# ============================================================

def estimate_freq_offset_4th_power(x, fs):
    """
    Rough frequency offset estimation.
    Works reasonably for QPSK-like signals.
    """

    y = x ** 4
    phase = np.unwrap(np.angle(y))
    n = np.arange(len(phase))

    slope, _ = np.polyfit(n, phase, 1)

    f_est = slope * fs / (2 * np.pi * 4)
    return f_est


def correct_freq_offset(x, fs, f_est):
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * f_est * n / fs)


# ============================================================
# SYMBOL EXTRACTION
# ============================================================

def get_symbols(x, sps, timing_offset):
    return x[timing_offset::sps]


# ============================================================
# DEVM CALCULATION
# ============================================================

def calculate_devm(symbols, ideal_diff):
    """
    Bluetooth EDR DEVM calculation.

    Measured differential symbol:
        d[n] = r[n] * conj(r[n-1])

    Compare d[n] with ideal differential pi/4-DQPSK symbol.
    """

    symbols = np.asarray(symbols)

    d_meas = symbols[1:] * np.conj(symbols[:-1])

    # Normalize differential amplitude
    d_meas = d_meas / np.maximum(np.abs(d_meas), 1e-12)

    n = min(len(d_meas), len(ideal_diff))
    d_meas = d_meas[:n]
    d_ideal = ideal_diff[:n]

    err_vec = d_meas - d_ideal

    devm = np.abs(err_vec) / np.maximum(np.abs(d_ideal), 1e-12) * 100.0

    rms_devm = np.sqrt(np.mean(devm ** 2))
    peak_devm = np.max(devm)
    devm_99 = np.percentile(devm, 99)

    return rms_devm, peak_devm, devm_99, devm, d_meas, d_ideal


# ============================================================
# FIND BEST PRBS9 ALIGNMENT
# ============================================================

def find_best_alignment(symbols, seed=0x1FF):
    """
    Brute-force search:
    - timing handled outside
    - PRBS9 shift
    - bit inversion
    - bit pair swap
    - mapping convention
    - IQ conjugation/sign options
    """

    best = None

    measured_versions = {
        "normal": symbols,
        "conjugate": np.conj(symbols),
        "negative": -symbols,
        "negative_conjugate": -np.conj(symbols),
    }

    num_meas_diff = len(symbols) - 1

    # Generate long PRBS9 stream.
    # 511-bit PRBS period. Need enough bits to cover capture + shifts.
    num_bits_needed = max(20000, 2 * num_meas_diff + 2000)
    prbs_bits = generate_prbs9_bits(num_bits_needed, seed=seed)

    for meas_name, sym_used in measured_versions.items():
        for invert_bits in [False, True]:
            for swap_bit_pair in [False, True]:
                for mapping_id in range(4):

                    ideal_long = bits_to_pi4dqpsk_diff_symbols(
                        prbs_bits,
                        mapping_id=mapping_id,
                        invert_bits=invert_bits,
                        swap_bit_pair=swap_bit_pair
                    )

                    # Try symbol shifts.
                    # 1022 bits = 511 EDR2M symbols often appears as repeat.
                    max_shift = min(1022, len(ideal_long) - num_meas_diff - 1)

                    for shift in range(max_shift):
                        ideal_shifted = ideal_long[shift:shift + num_meas_diff]

                        rms, peak, p99, err, d_meas, d_ideal = calculate_devm(
                            sym_used,
                            ideal_shifted
                        )

                        if best is None or rms < best["rms"]:
                            best = {
                                "rms": rms,
                                "peak": peak,
                                "p99": p99,
                                "err": err,
                                "d_meas": d_meas,
                                "d_ideal": d_ideal,
                                "symbols": sym_used,
                                "measured_version": meas_name,
                                "invert_bits": invert_bits,
                                "swap_bit_pair": swap_bit_pair,
                                "mapping_id": mapping_id,
                                "shift": shift,
                            }

    return best


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading IQ...")
    x = load_iq_csv(csv_file)

    sps_float = fs / symbol_rate
    sps = int(round(sps_float))

    print(f"IQ samples        : {len(x)}")
    print(f"Sample rate       : {fs/1e6:.6f} MS/s")
    print(f"Symbol rate       : {symbol_rate/1e6:.6f} Msymbol/s")
    print(f"Samples/symbol    : {sps_float:.6f}")

    if abs(sps_float - sps) > 1e-6:
        raise ValueError(
            "fs / symbol_rate is not integer. "
            "This simple code needs integer SPS. Resample first."
        )

    print("\nEstimating frequency offset...")
    f_est = estimate_freq_offset_4th_power(x, fs)
    print(f"Estimated freq offset: {f_est:.2f} Hz")

    x_corr = correct_freq_offset(x, fs, f_est)

    global_best = None

    print("\nSearching timing offset + PRBS9 alignment...")
    for timing_offset in range(sps):
        symbols = get_symbols(x_corr, sps, timing_offset)

        if len(symbols) < 500:
            continue

        best = find_best_alignment(symbols, seed=prbs9_seed)
        best["timing_offset"] = timing_offset

        print(
            f"Timing {timing_offset:2d}/{sps}: "
            f"best RMS DEVM = {best['rms']:.2f}%"
        )

        if global_best is None or best["rms"] < global_best["rms"]:
            global_best = best

    if global_best is None:
        raise RuntimeError("No valid DEVM result found.")

    print("\n====================================================")
    print("Bluetooth EDR2M Payload-only PRBS9 DEVM Result")
    print("====================================================")
    print(f"PRBS9 seed          : 0x{prbs9_seed:X}")
    print(f"Timing offset       : {global_best['timing_offset']} / {sps}")
    print(f"Measured version    : {global_best['measured_version']}")
    print(f"Mapping ID          : {global_best['mapping_id']}")
    print(f"Invert bits         : {global_best['invert_bits']}")
    print(f"Swap bit pair       : {global_best['swap_bit_pair']}")
    print(f"PRBS symbol shift   : {global_best['shift']}")
    print("----------------------------------------------------")
    print(f"RMS DEVM            : {global_best['rms']:.2f} %")
    print(f"Peak DEVM           : {global_best['peak']:.2f} %")
    print(f"99% DEVM            : {global_best['p99']:.2f} %")
    print("----------------------------------------------------")
    print(f"RMS limit           : {LIMIT_RMS:.2f} %")
    print(f"Peak limit          : {LIMIT_PEAK:.2f} %")
    print(f"99% limit           : {LIMIT_99:.2f} %")

    if (
        global_best["rms"] <= LIMIT_RMS
        and global_best["peak"] <= LIMIT_PEAK
        and global_best["p99"] <= LIMIT_99
    ):
        print("Result              : PASS-like")
    else:
        print("Result              : FAIL-like")

    # ========================================================
    # PLOTS
    # ========================================================

    symbols = global_best["symbols"]
    d_meas = global_best["d_meas"]
    d_ideal = global_best["d_ideal"]
    err = global_best["err"]

    plt.figure()
    plt.scatter(symbols.real, symbols.imag, s=4)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Recovered Symbol Constellation")
    plt.xlabel("I")
    plt.ylabel("Q")

    plt.figure()
    plt.scatter(d_meas.real, d_meas.imag, s=4, label="Measured differential")
    plt.scatter(d_ideal.real, d_ideal.imag, s=30, marker="x", label="Ideal differential")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Differential Constellation")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.legend()

    plt.figure()
    plt.plot(err)
    plt.grid(True)
    plt.title("DEVM per Symbol")
    plt.xlabel("Symbol index")
    plt.ylabel("DEVM (%)")

    plt.show()


if __name__ == "__main__":
    main()
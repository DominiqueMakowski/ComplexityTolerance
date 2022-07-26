import neurokit2 as nk
import numpy as np
import pandas as pd

# Parameters
# df = pd.read_csv("data_Signals.csv")

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "Random-Walk"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), color="red", show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "lorenz_10_2.5_28"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "lorenz_20_2_30"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "oscillatory"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "fractal"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=3, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)


# ================
# Generate Signal
# ================
def run_benchmark(noise_intensity=0.01):
    # Initialize data storage
    data_signal = []
    data_tolerance = []

    print("Noise intensity: {}".format(noise_intensity))
    for duration in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for method in [
            "Random-Walk",
            "lorenz_10_2.5_28",
            "lorenz_20_2_30",
            "oscillatory",
            "fractal",
            "EEG",
        ]:
            if method == "Random-Walk":
                delay = 10
                signal = nk.complexity_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    method="random",
                )
            elif method == "lorenz_10_2.5_28":
                delay = 4
                signal = nk.complexity_simulate(
                    duration=duration * 2,
                    sampling_rate=500,
                    method="lorenz",
                    sigma=10.0,
                    beta=2.5,
                    rho=28.0,
                )
            elif method == "lorenz_20_2_30":
                delay = 15
                signal = nk.complexity_simulate(
                    duration=duration * 2,
                    sampling_rate=500,
                    method="lorenz",
                    sigma=20.0,
                    beta=2,
                    rho=30.0,
                )

            elif method == "oscillatory":
                delay = 10
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[2, 5, 11, 18, 24, 42, 60, 63],
                )
            elif method == "fractal":
                delay = 5
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[4, 8, 16, 32, 64],
                    amplitude=[2, 2, 1, 1, 0.5],
                )
            elif method == "EEG":
                delay = 20
                k = 20
                signal = nk.eeg_simulate(
                    duration=duration,
                    sampling_rate=1000,
                )

            # Standardize
            signal = nk.standardize(signal)

            # Add Noise
            for noise in np.linspace(-2, 2, 5):
                noise_ = nk.signal_noise(duration=duration, sampling_rate=1000, beta=noise)
                signal_ = nk.standardize(signal + (nk.standardize(noise_) * noise_intensity))

                # Save the signal to visualize the type of signals fed into the benchmarking
                if duration == 1:

                    data_signal.append(
                        pd.DataFrame(
                            {
                                "Signal": signal_,
                                "Length": len(signal_),
                                "Duration": range(1, len(signal_) + 1),
                                "Noise": noise,
                                "Noise_Intensity": noise_intensity,
                                "Method": method,
                            }
                        )
                    )

                r_range = np.linspace(0.02, 2, 50)

                for m in range(1, 10):

                    r1, info = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="maxApEn",
                    )
                    rez = pd.DataFrame({"Tolerance": info["Values"], "Score": info["Scores"]})
                    rez["Method"] = "Approximate Entropy"

                    _, info = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="neighbours",
                    )
                    temp = pd.DataFrame({"Tolerance": info["Values"], "Score": info["Scores"]})
                    temp["Method"] = "Nearest Neighbours"
                    rez = pd.concat([rez, temp], axis=0)

                    _, info = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="recurrence",
                    )
                    temp = pd.DataFrame({"Tolerance": info["Values"], "Score": info["Scores"]})
                    temp["Method"] = "Recurrence Rate"
                    rez = pd.concat([rez, temp], axis=0)

                    r2, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="sd",
                    )
                    r3, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="nolds",
                    )
                    r4, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="chon2009",
                    )

                    # Add info
                    rez["Optimal_maxApEn"] = r1
                    rez["Optimal_SD"] = r2
                    rez["Optimal_Scholzel"] = r3
                    rez["Optimal_Chon"] = r4
                    rez["Length"] = len(signal_)
                    rez["Noise_Type"] = noise
                    rez["Noise_Intensity"] = noise_intensity
                    rez["Signal"] = method
                    rez["Dimension"] = m

                    data_tolerance.append(rez)
    return pd.concat(data_signal), pd.concat(data_tolerance)


# run_benchmark(noise_intensity=0.01)
out = nk.parallel_run(
    run_benchmark,
    [{"noise_intensity": i} for i in np.linspace(0.001, 3, 32)],
    n_jobs=32,
    verbose=10,
)

# pd.concat([out[i][0] for i in range(len(out))]).to_csv("data_Signals.csv", index=False)
# pd.concat([out[i][1] for i in range(len(out))]).to_csv("data_Tolerance.csv", index=False)
df = pd.concat([out[i][1] for i in range(len(out))])
nk.write_csv(df, "data/data_Tolerance", parts=30)

print("FINISHED.")

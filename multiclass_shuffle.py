import random
import time
from pathlib import Path

import numpy as np
import uproot


# ============================================================
# Configuration
# ============================================================

mc_dir = Path("/volatile/clas12/zurek/SULI/mc_v01")

output_dir = Path("slurm/multiclass")
output_dir.mkdir(parents=True, exist_ok=True)

seed = 42

# ============================================================
# Find ROOT files
# ============================================================

files = sorted(mc_dir.glob("*.root"))

print(f"Found {len(files)} ROOT files")
print()


# ============================================================
# Scan for proton content
# ============================================================

proton_files = []
other_files = []

print("Scanning ROOT files for proton content...")

start_time = time.time()
total_files = len(files)

for i, f in enumerate(files, start=1):

    try:
        with uproot.open(f) as root:

            tree_key = root.keys()[0]
            tree = root[tree_key]

            pid = tree["mc_matching_pid"].array(
                library="np"
            )

        if np.any(pid == 2212):
            proton_files.append(f.name)
            has_proton = True
        else:
            other_files.append(f.name)
            has_proton = False


    except Exception as e:
        print(
            f"WARNING: failed reading {f.name}: {e}"
        )
        continue


    # Progress report
    elapsed = time.time() - start_time

    rate = i / elapsed if elapsed > 0 else 0
    remaining = (
        (total_files - i) / rate
        if rate > 0
        else 0
    )

    print(
        f"[{i}/{total_files}] "
        f"{f.name} | "
        f"{'PROTON' if has_proton else 'no proton'} | "
        f"elapsed {elapsed/60:.1f} min | "
        f"ETA {remaining/60:.1f} min"
    )


scan_time = (time.time() - start_time) / 60

print()
print(
    f"Scan complete in {scan_time:.2f} minutes"
)

print(
    f"Files with protons: {len(proton_files)}"
)

print(
    f"Files without protons: {len(other_files)}"
)

print()


# ============================================================
# Save proton file list
# ============================================================

with open(output_dir / "proton_files.txt", "w") as f:
    f.write(
        "\n".join(sorted(proton_files))
        + "\n"
    )


# ============================================================
# Shuffle
# ============================================================

random.seed(seed)

random.shuffle(proton_files)
random.shuffle(other_files)


# ============================================================
# Split function
# ============================================================

def split_files(file_list):

    n = len(file_list)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train = file_list[:n_train]
    val = file_list[n_train:n_train+n_val]
    test = file_list[n_train+n_val:]

    return train, val, test



# Split proton and non-proton files separately
#
# This guarantees proton-containing files
# appear in all three splits.

p_train, p_val, p_test = split_files(proton_files)

o_train, o_val, o_test = split_files(other_files)


# Combine

train = p_train + o_train
val = p_val + o_val
test = p_test + o_test


# Shuffle final splits

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)


# ============================================================
# Save split files
# ============================================================

splits = {
    "train_m": train,
    "val_m": val,
    "test_m": test,
}


print("Final split sizes:")
print()

for name, split in splits.items():

    print(
        f"{name}: {len(split)} files"
    )

    with open(
        output_dir / f"{name}_files.txt",
        "w",
    ) as f:

        f.write(
            "\n".join(sorted(split))
            + "\n"
        )


# ============================================================
# Save summary
# ============================================================

summary_path = output_dir / "scan_summary.txt"

with open(summary_path, "w") as f:

    f.write(
        f"Total files: {len(files)}\n"
    )

    f.write(
        f"Proton files: {len(proton_files)}\n"
    )

    f.write(
        f"Non-proton files: {len(other_files)}\n\n"
    )

    for name, split in splits.items():

        f.write(
            f"{name}: {len(split)} files\n"
        )

    f.write("\n")

    f.write(
        f"Proton distribution:\n"
    )

    f.write(
        f"  train: {len(p_train)}\n"
    )

    f.write(
        f"  val:   {len(p_val)}\n"
    )

    f.write(
        f"  test:  {len(p_test)}\n"
    )


print()
print("Done.")
print(
    f"Saved splits to: {output_dir}"
)
print(
    f"Summary saved to: {summary_path}"
)
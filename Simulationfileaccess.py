import uproot
from tqdm import trange
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages
import re

#going through multiple files
root_files = sorted(glob.glob(os.path.join("../", "AmBe_Neutron_*.root")))

with PdfPages("All_ClusterPE_vs_ChargeBalance.pdf") as pdf:
    for file in root_files:
        print(f"Processing: {file}")

        base = os.path.basename(file)
        match = re.search(r'AmBe_Neutron_(.*)\.root', base)
        tag = match.group(1) if match else "Unknown"
        file = uproot.open(file)


        tree = file["phaseIITankClusterTree"]
        print(tree.keys())  # Lists all branches in the tree

        data = tree.arrays(["clusterPE", "clusterChargeBalance"], library="np")
        cluster_pe = data["clusterPE"]
        cluster_charge_balance = data["clusterChargeBalance"]

        plt.figure(figsize=(8, 6))
        plt.hist2d(cluster_pe, cluster_charge_balance, bins=50, cmap="viridis")
        plt.colorbar(label="Counts")
        plt.xlabel("clusterPE")
        plt.ylabel("clusterChargeBalance")
        plt.title(f"clusterPE vs clusterChargeBalance for {tag}")
        pdf.savefig()
        #plt.savefig(f"clusterPE_vs_chargeBalance_{tag}.png", dpi=300)
        plt.close()
    

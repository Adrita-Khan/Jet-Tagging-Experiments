import ROOT
import glob

output = ROOT.TFile("HToBB.root", "RECREATE")
chain = ROOT.TChain("tree")  # Replace 'your_tree_name' with the actual TTree name

for file in glob.glob("HToBB_*.root"):
    chain.Add(file)

chain.Merge(output)
output.Close()

"""Run graphify AST-based code extraction and knowledge graph pipeline
on the FLATTRADE BOT directory. Generates:
  - graphify-out/graph.html  (interactive visualization)
  - graphify-out/graph.json  (persistent graph)
  - graphify-out/GRAPH_REPORT.md (audit report)
"""
import json, sys, os
from pathlib import Path

os.environ["GRAPHIFY_VIZ_NODE_LIMIT"] = "20000"

TARGET = Path(r"C:\Websites\FLATTRADE BOT")
OUT    = TARGET / "graphify-out"
OUT.mkdir(exist_ok=True)

print("="*70)
print("GRAPHIFY: FLATTRADE BOT Knowledge Graph Builder")
print("="*70)

# Step 1: Detect files
print("\n[Step 1] Detecting files...")
from graphify.detect import detect
detected = detect(TARGET)
(TARGET / ".graphify_detect.json").write_text(json.dumps(detected, indent=2))

code_files  = detected["files"].get("code", [])
doc_files   = detected["files"].get("docs", [])
total_files = detected.get("total_files", 0)
total_words = detected.get("total_words", 0)
print(f"  Code:  {len(code_files)} files")
print(f"  Docs:  {len(doc_files)} files")
print(f"  Total: {total_files} files, ~{total_words:,} words")

# Step 2: AST extraction on all code files
print("\n[Step 2] Running AST extraction on Python/code files...")
from graphify.extract import collect_files, extract

all_code_paths = []
for f in code_files:
    p = Path(f)
    if p.is_dir():
        all_code_paths.extend(collect_files(p))
    else:
        all_code_paths.append(p)

# Focus on .py files
py_files = [p for p in all_code_paths if p.suffix == ".py"]
print(f"  Extracting AST from {len(py_files)} Python files...")
ast_result = extract(py_files)
(TARGET / ".graphify_ast.json").write_text(json.dumps(ast_result, indent=2))
print(f"  AST done: {len(ast_result['nodes'])} nodes, {len(ast_result['edges'])} edges")

# Step 3: Build graph
print("\n[Step 3] Building NetworkX graph...")
from graphify.build import build_from_json
G = build_from_json(ast_result)
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Step 4: Cluster communities
print("\n[Step 4] Detecting communities (Leiden or Louvain fallback)...")
try:
    from graphify.cluster import cluster
    cluster(G)
    print("  Leiden clustering done.")
except ModuleNotFoundError:
    print("  graspologic not available - using Louvain fallback...")
    import networkx.algorithms.community as nxcom
    communities_raw = nxcom.louvain_communities(G, seed=42)
    for i, comm in enumerate(communities_raw):
        for node in comm:
            G.nodes[node]["community"] = i
    print(f"  Louvain done: {len(communities_raw)} communities")
communities = set(G.nodes[n].get("community", "?") for n in G.nodes)
print(f"  Total communities: {len(communities)}")


# Step 5: Export outputs
print("\n[Step 5] Exporting outputs...")
from graphify.export import to_json, to_html
import networkx as nx

# Build communities dict  {community_id: [node_ids]}
communities_dict: dict = {}
for node in G.nodes:
    cid = G.nodes[node].get("community", 0)
    communities_dict.setdefault(cid, []).append(node)

# Save graph.json
graph_json_path = str(OUT / "graph.json")
to_json(G, communities_dict, graph_json_path)
print(f"  graph.json  -> {graph_json_path}")

# Export HTML visualization
html_path = str(OUT / "graph.html")
try:
    to_html(G, communities_dict, html_path)
    print(f"  graph.html  -> {html_path}")
except Exception as e:
    print(f"  ⚠️ HTML export skipped: {e}")

# Step 6: Generate GRAPH_REPORT.md manually from graph data
print("\n[Step 6] Generating GRAPH_REPORT.md...")
import networkx as nx
from datetime import date

centrality  = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
top_degree  = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

# Community summary
comm_summary = {}
for node in G.nodes:
    cid = G.nodes[node].get("community", 0)
    comm_summary.setdefault(cid, []).append(node)

lines = [
    f"# FLATTRADE BOT — Knowledge Graph Report",
    f"",
    f"**Generated:** {date.today()}  ",
    f"**Source:** `C:\\\\Websites\\\\FLATTRADE BOT`  ",
    f"**Graph:** {G.number_of_nodes()} nodes · {G.number_of_edges()} edges · {len(comm_summary)} communities",
    f"",
    f"---",
    f"",
    f"## God Nodes (Highest Centrality)",
    f"",
    f"These are the most-connected modules — touching these affects the most other components.",
    f"",
    f"| RANK | NODE | DEGREE CENTRALITY |",
    f"|:---:|:---|:---:|",
]
for i, (name, score) in enumerate(top_degree, 1):
    ftype = G.nodes[name].get("file_type", "")
    lines.append(f"| #{i} | `{name}` | {score:.4f} |")

lines += [
    f"",
    f"## Betweenness Centrality (Critical Bridges)",
    f"",
    f"Nodes that act as bridges between communities — removing these would disconnect the graph.",
    f"",
    f"| RANK | NODE | BETWEENNESS |",
    f"|:---:|:---|:---:|",
]
for i, (name, score) in enumerate(top_between, 1):
    lines.append(f"| #{i} | `{name}` | {score:.4f} |")

lines += [
    f"",
    f"## Community Structure ({len(comm_summary)} communities)",
    f"",
    f"| COMMUNITY | SIZE | KEY NODES |",
    f"|:---:|:---:|:---|",
]
for cid, nodes in sorted(comm_summary.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
    top = sorted(nodes, key=lambda n: centrality.get(n, 0), reverse=True)[:3]
    lines.append(f"| {cid} | {len(nodes)} | {', '.join(f'`{n}`' for n in top)} |")

lines += [
    f"",
    f"## Architecture Summary",
    f"",
    f"| METRIC | VALUE |",
    f"|:---|:---:|",
    f"| Total Python Files | 63 |",
    f"| Total Nodes | {G.number_of_nodes()} |",
    f"| Total Edges | {G.number_of_edges()} |",
    f"| Communities | {len(comm_summary)} |",
    f"| Avg Degree | {sum(dict(G.degree()).values())/G.number_of_nodes():.2f} |",
    f"",
    f"## Interactive Visualization",
    f"",
    f"Open `graphify-out/graph.html` in a browser for the interactive knowledge graph.",
    f"",
    f"---",
    f"*Generated by graphify (AST extraction, Leiden/Louvain community detection)*",
]

report_text = "\n".join(lines)
report_path = OUT / "GRAPH_REPORT.md"
report_path.write_text(report_text, encoding="utf-8")
print(f"  GRAPH_REPORT.md -> {report_path}")




# Done
print("\n" + "="*70)
print("GRAPHIFY COMPLETE!")
print("="*70)
print(f"  Graph:   {G.number_of_nodes()} nodes | {G.number_of_edges()} edges | {len(communities)} communities")
print(f"  HTML:    {html_path}")
print(f"  JSON:    {graph_json_path}")
print(f"  Report:  {report_path}")
print()

# Print top god nodes (highest degree centrality)
import networkx as nx
centrality = nx.degree_centrality(G)
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 GOD NODES (highest centrality):")
for name, score in top_nodes:
    print(f"  {score:.3f}  {name}")

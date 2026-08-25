# RakshaPay Transaction Graph Structure & Network Analysis Documentation

Comprehensive guide to the **Graph Data Model**, **Transaction Network Traversal**, **Money Mule Ring Detection**, and **Graph API Endpoints** for RakshaPay.

---

## 📑 Table of Contents
1. [Overview & Fraud Detection Architecture](#1-overview--fraud-detection-architecture)
2. [Graph Data Model (Nodes & Edges Specification)](#2-graph-data-model-nodes--edges-specification)
3. [Graph Algorithms for Fraud Detection](#3-graph-algorithms-for-fraud-detection)
4. [Backend Graph API Endpoints](#4-backend-graph-api-endpoints)
5. [Database Graph Queries (PostgreSQL Recursive CTEs)](#5-database-graph-queries-postgresql-recursive-ctes)
6. [Graph Database Integration (Neo4j / Apache AGE)](#6-graph-database-integration-neo4j--apache-age)
7. [Frontend Interactive Visualization Specification](#7-frontend-interactive-visualization-specification)

---

## 1. Overview & Fraud Detection Architecture

In modern financial risk intelligence, isolated transaction scoring is insufficient to catch sophisticated money mule networks, smurfing, and circular layering schemes.

RakshaPay models the entire transaction ecosystem as a **Directed Attributed Multi-Graph**:
* **$G = (V, E)$** where:
  * **$V$ (Vertices / Nodes)**: Bank accounts, customer identities, devices, and UPI handles.
  * **$E$ (Edges / Links)**: Directed monetary transactions, device sharing links, and IP co-locations.

```
 ┌──────────────┐         ₹4,50,000 (IMPS)         ┌──────────────┐
 │  Account A   │ ───────────────────────────────► │  Account B   │ (Mule Aggregator)
 │   (Source)   │                                  │ (Risk: 88%)  │
 └──────┬───────┘                                  └──────┬───────┘
        │                                                 │
        │ ₹3,20,000 (UPI)                                 │ ₹7,50,000 (RTGS)
        ▼                                                 ▼
 ┌──────────────┐         ₹7,00,000 (NEFT)         ┌──────────────┐
 │  Account C   │ ───────────────────────────────► │  Account D   │ (Mule Exit / Cash-out)
 │ (Risk: 72%)  │                                  │ (Risk: 94%)  │
 └──────────────┘                                  └──────────────┘
```

---

## 2. Graph Data Model (Nodes & Edges Specification)

### A. Node Schema (Vertices)

```json
{
  "id": "rp-092",
  "type": "ACCOUNT",
  "label": "Rahul Sharma",
  "properties": {
    "accountNumber": "0923 5678 9012",
    "customerId": "CUST-778899",
    "riskScore": 64,
    "riskLevel": "Medium",
    "branch": "Connaught Place Branch",
    "balance": 684220.00,
    "kycStatus": "Verified",
    "isHeld": false
  }
}
```

### B. Edge Schema (Transactions & Relationships)

```json
{
  "id": "TXN-90214",
  "type": "TRANSACTION",
  "source": "rp-092",
  "target": "rp-138",
  "properties": {
    "amount": 185000.00,
    "currency": "INR",
    "channel": "IMPS",
    "timestamp": "2026-08-25T10:14:00Z",
    "riskScore": 81,
    "status": "Cleared",
    "velocityHop": 2
  }
}
```

---

## 3. Graph Algorithms for Fraud Detection

| Algorithm | Fraud Pattern Detected | How It Works |
| :--- | :--- | :--- |
| **Cycle Detection (Tarjan / DFS)** | **Circular Money Laundering** | Detects closed loops where money returns to the source account ($A \rightarrow B \rightarrow C \rightarrow A$). |
| **K-Hop Neighborhood Search** | **Guilt by Association** | Traces all accounts within $k$ steps ($k=1, 2, 3$) of a known high-risk/blocked account. |
| **In/Out Degree Centrality** | **Mule Aggregators & Dispersion Hubs** | Flags accounts with abnormally high fan-in (many small deposits) followed by immediate fan-out (single large transfer). |
| **Shortest Flow Path (Dijkstra / BFS)** | **Rapid Hop Tracing** | Identifies the fastest path through which stolen funds traversed multiple intermediary accounts. |
| **Community Detection (Louvain)** | **Organized Fraud Rings** | Groups densely interconnected suspicious accounts operating in coordination. |

---

## 4. Backend Graph API Endpoints

**Base URL**: `http://localhost:3000/api/graph`

---

### 1. Get Account 2-Hop Network Graph
* **Method**: `GET`
* **Route**: `/api/graph/accounts/:id/network`
* **Query Parameters**:
  * `depth` *(integer, default: 2, max: 4)* — Number of traversal hops.
  * `minAmount` *(number, optional)* — Minimum transaction threshold.
* **Success Response (`200 OK`)**:
  ```json
  {
    "accountId": "rp-092",
    "depth": 2,
    "graph": {
      "nodes": [
        {
          "id": "rp-092",
          "label": "Rahul Sharma",
          "riskScore": 64,
          "riskLevel": "Medium",
          "isRoot": true
        },
        {
          "id": "rp-138",
          "label": "Rakesh Kumar",
          "riskScore": 82,
          "riskLevel": "High",
          "isRoot": false
        },
        {
          "id": "rp-369",
          "label": "Vikram Patel",
          "riskScore": 91,
          "riskLevel": "High",
          "isRoot": false
        }
      ],
      "edges": [
        {
          "id": "TXN-90214",
          "source": "rp-092",
          "target": "rp-138",
          "amount": "₹1,85,000",
          "channel": "IMPS",
          "timestamp": "10:14 AM",
          "risk": "High"
        },
        {
          "id": "TXN-90355",
          "source": "rp-138",
          "target": "rp-369",
          "amount": "₹3,40,000",
          "channel": "RTGS",
          "timestamp": "10:32 AM",
          "risk": "High"
        }
      ]
    },
    "metrics": {
      "totalNodes": 3,
      "totalEdges": 2,
      "maxRiskInCluster": 91,
      "suspectedMuleRing": true
    }
  }
  ```

---

### 2. Detect Money Mule Rings & Cycles
* **Method**: `GET`
* **Route**: `/api/graph/mule-rings`
* **Query Parameters**:
  * `maxCycleLength` *(integer, default: 5)* — Max hops in circular loop.
* **Success Response (`200 OK`)**:
  ```json
  {
    "detectedRingsCount": 2,
    "rings": [
      {
        "ringId": "RING-001",
        "cycleLength": 3,
        "totalValue": "₹12,45,000",
        "accountsInvolved": ["rp-092", "rp-138", "rp-369"],
        "confidence": 0.94,
        "pattern": "Circular Layering Scheme",
        "path": "Rahul Sharma -> Rakesh Kumar -> Vikram Patel -> Rahul Sharma"
      }
    ]
  }
  ```

---

### 3. Trace Shortest Money Flow Path
* **Method**: `GET`
* **Route**: `/api/graph/path`
* **Query Parameters**:
  * `source`: Origin account ID (`rp-092`).
  * `target`: Destination account ID (`rp-369`).
* **Success Response (`200 OK`)**:
  ```json
  {
    "source": "rp-092",
    "target": "rp-369",
    "hopCount": 2,
    "path": [
      {
        "step": 1,
        "from": "rp-092 (Rahul Sharma)",
        "to": "rp-138 (Rakesh Kumar)",
        "transactionId": "TXN-90214",
        "amount": "₹1,85,000",
        "timestamp": "10:14 AM"
      },
      {
        "step": 2,
        "from": "rp-138 (Rakesh Kumar)",
        "to": "rp-369 (Vikram Patel)",
        "transactionId": "TXN-90355",
        "amount": "₹3,40,000",
        "timestamp": "10:32 AM"
      }
    ]
  }
  ```

---

## 5. Database Graph Queries (PostgreSQL Recursive CTEs)

PostgreSQL natively supports fast graph traversals using `WITH RECURSIVE`:

### Multi-Hop Money Trail Query:
```sql
WITH RECURSIVE MoneyTrail AS (
  -- Base case: Immediate transactions from initial account
  SELECT 
    t.id AS transaction_id,
    t.account_id AS source_account,
    t.counterparty AS target_account,
    t.amount,
    t.timestamp,
    1 AS hop_level,
    ARRAY[t.account_id::text] AS path
  FROM transactions t
  WHERE t.account_id = 'rp-092'

  UNION ALL

  -- Recursive step: Follow downstream transfers
  SELECT 
    t.id,
    t.account_id,
    t.counterparty,
    t.amount,
    t.timestamp,
    mt.hop_level + 1,
    mt.path || t.account_id::text
  FROM transactions t
  JOIN MoneyTrail mt ON t.account_id = mt.target_account
  WHERE mt.hop_level < 3 -- Limit depth to 3 hops
    AND NOT (t.account_id::text = ANY(mt.path)) -- Prevent infinite cycles
)
SELECT * FROM MoneyTrail ORDER BY hop_level, timestamp;
```

---

## 6. Graph Database Integration (Neo4j / Apache AGE)

If scaling to millions of accounts, RakshaPay can synchronize transactions to a **Neo4j** or **Apache AGE** graph engine.

### Cypher Query for Detecting 3-Node Mule Rings:
```cypher
MATCH path = (a:Account)-[:TRANSFERRED]->(b:Account)-[:TRANSFERRED]->(c:Account)-[:TRANSFERRED]->(a:Account)
WHERE a <> b AND b <> c AND a <> c
RETURN path, 
       reduce(total = 0, r IN relationships(path) | total + r.amount) AS totalVolume,
       nodes(path) AS muleRingMembers
ORDER BY totalVolume DESC;
```

---

## 7. Frontend Interactive Visualization Specification

When displaying the transaction graph in the admin dashboard:

1. **Libraries**: `Cytoscape.js`, `D3.js` (Force-Directed Graph), or `React Flow`.
2. **Node Visual Encoding**:
   * **Node Color**:
     * 🟢 Low Risk (Score $\le 40$)
     * 🟡 Medium Risk (Score $41 - 70$)
     * 🔴 High Risk (Score $> 70$)
   * **Node Size**: Proportional to total transaction volume processed.
3. **Edge Visual Encoding**:
   * **Arrow Direction**: Flow of funds (Debit $\rightarrow$ Credit).
   * **Edge Thickness**: Transaction amount.
   * **Edge Color**: Red if transaction risk is High/Blocked.
4. **Interactive Features**:
   * Click on any node to expand $k$-hop connections.
   * Hover over edge to view transaction ID, timestamp, and amount.
   * "Isolate Subgraph" button to view suspected mule rings.

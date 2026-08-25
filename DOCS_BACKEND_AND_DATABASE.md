# RakshaPay Backend API & Database Documentation

Comprehensive guide for configuring, connecting, and extending the **RakshaPay Backend API Server** and **PostgreSQL Database** with Drizzle ORM.

---

## 📑 Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Environment Configuration (`.env`)](#2-environment-configuration-env)
3. [Database Setup & Connection](#3-database-setup--connection)
4. [Database Schema & Models (Drizzle ORM)](#4-database-schema--models-drizzle-orm)
5. [Backend API Endpoints](#5-backend-api-endpoints)
6. [Frontend & Client Integration](#6-frontend--client-integration)
7. [ML Model & Risk Engine Integration](#7-ml-model--risk-engine-integration)
8. [CLI Commands Reference](#8-cli-commands-reference)

---

## 1. Architecture Overview

```
 ┌───────────────────────────┐      ┌───────────────────────────┐
 │   RakshaPay Admin / User   │      │     ML Risk Pipeline      │
 │  Frontend (React / Vite)  │      │   (Python / LightGBM /    │
 │   http://localhost:5173   │      │      AutoEncoder)         │
 └─────────────┬─────────────┘      └─────────────┬─────────────┘
               │ HTTP / JSON                      │ Predict / Score
               ▼                                  ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                Express.js API Server (ESM)                   │
 │             artifacts/api-server (Port 3000)                 │
 │  - Pino Logging    - CORS Enabled    - Zod Validation Body   │
 └──────────────────────────────┬───────────────────────────────┘
                                │ Connection Pool (node-postgres)
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                  PostgreSQL Database Layer                   │
 │                      lib/db (Drizzle ORM)                    │
 │  - Accounts        - Transactions    - Risk Alerts           │
 └──────────────────────────────────────────────────────────────┘
```

### Monorepo Components:
* **API Server (`artifacts/api-server`)**: Express.js HTTP service listening on `/api`.
* **Database Layer (`lib/db`)**: PostgreSQL connection pool and Drizzle ORM schemas.
* **Validation Layer (`lib/api-zod`)**: Runtime schema validation using Zod.
* **API Spec (`lib/api-spec`)**: OpenAPI 3.1 contract and Orval generators.
* **React API Client (`lib/api-client-react`)**: TanStack Query hooks.
* **ML Intelligence (`ml/`)**: Machine learning models for fraud and anomaly detection.

---

## 2. Environment Configuration (`.env`)

Create a `.env` file in the project root or in `artifacts/api-server/.env`:

```env
# ==========================================
# API Server Configuration
# ==========================================
PORT=3000
NODE_ENV=development
BASE_PATH=/api

# ==========================================
# Database Configuration (PostgreSQL)
# ==========================================
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/rakshapay

# Examples for cloud-hosted PostgreSQL:
# Neon:     postgresql://user:pass@ep-cool-xyz.us-east-1.aws.neon.tech/rakshapay?sslmode=require
# Supabase: postgresql://postgres:pass@db.xxxx.supabase.co:5432/postgres
# AWS RDS:  postgresql://master:pass@rakshapay.xyz.rds.amazonaws.com:5432/rakshapay

# ==========================================
# Frontend CORS Allowed Origins
# ==========================================
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
```

---

## 3. Database Setup & Connection

The database connection is managed via [lib/db/src/index.js](file:///Users/pshreetam/Desktop/rakshapay-admin-main/lib/db/src/index.js) using `pg.Pool` and Drizzle ORM:

```javascript
import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "./schema/index.js";

const { Pool } = pg;

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL must be set. Did you forget to provision a database?");
}

export const pool = new Pool({ 
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === "production" ? { rejectUnauthorized: false } : undefined,
});

export const db = drizzle(pool, { schema });
export * from "./schema/index.js";
```

---

## 4. Database Schema & Models (Drizzle ORM)

Located in [lib/db/src/schema/](file:///Users/pshreetam/Desktop/rakshapay-admin-main/lib/db/src/schema/):

### Core Tables Schema Definition:

```javascript
import { pgTable, serial, text, integer, timestamp, numeric, boolean } from "drizzle-orm/pg-core";

// 1. Accounts Table
export const accountsTable = pgTable("accounts", {
  id: text("id").primaryKey(), // e.g. 'rp-092'
  holder: text("holder").notNull(),
  accountNumber: text("account_number").notNull().unique(),
  customerId: text("customer_id").notNull(),
  ifscCode: text("ifsc_code").default("RKSH0000108"),
  branchCode: text("branch_code").default("CP-0108"),
  type: text("type").notNull(), // 'Savings' | 'Current'
  kyc: text("kyc").default("Verified"), // 'Verified' | 'Review' | 'Pending'
  score: integer("score").default(50), // 0 to 100
  risk: text("risk").default("Low"), // 'Low' | 'Medium' | 'High' | 'Under review'
  balance: numeric("balance", { precision: 14, scale: 2 }).default("0.00"),
  branch: text("branch").default("Connaught Place Branch"),
  phone: text("phone"),
  email: text("email"),
  isHeld: boolean("is_held").default(false),
  openedAt: timestamp("opened_at").defaultNow(),
});

// 2. Transactions Table
export const transactionsTable = pgTable("transactions", {
  id: text("id").primaryKey(), // e.g. 'TXN-90214'
  accountId: text("account_id").references(() => accountsTable.id),
  customer: text("customer").notNull(),
  counterparty: text("counterparty").notNull(),
  channel: text("channel").notNull(), // 'UPI', 'NEFT', 'IMPS', 'RTGS', 'Card'
  amount: numeric("amount", { precision: 14, scale: 2 }).notNull(),
  type: text("type").notNull(), // 'Debit' | 'Credit'
  risk: text("risk").default("Low"), // 'Low' | 'Medium' | 'High'
  status: text("status").default("Cleared"), // 'Cleared' | 'Review' | 'Blocked'
  timestamp: timestamp("timestamp").defaultNow(),
});

// 3. Risk Alerts Table
export const alertsTable = pgTable("alerts", {
  id: text("id").primaryKey(), // e.g. 'ALT-408'
  accountId: text("account_id").references(() => accountsTable.id),
  account: text("account").notNull(),
  title: text("title").notNull(),
  detail: text("detail").notNull(),
  severity: text("severity").notNull(), // 'Critical' | 'High' | 'Medium'
  status: text("status").default("Open"), // 'Open' | 'Investigating' | 'Resolved'
  createdAt: timestamp("created_at").defaultNow(),
});
```

---

## 5. Backend API Endpoints

Base URL: `http://localhost:3000/api`

### 1. Health & Status
* **`GET /api/healthz`**
  * **Summary**: Check server health.
  * **Response (`200 OK`)**:
    ```json
    {
      "status": "ok"
    }
    ```

---

### 2. Accounts Endpoints

* **`GET /api/accounts`**
  * **Summary**: Get paginated list of bank accounts with search and risk filters.
  * **Query Parameters**:
    * `search` *(optional, string)*: Filter by customer name, account number, or customer ID.
    * `risk` *(optional, string)*: Filter by risk level (`Low`, `Medium`, `High`, `Under review`).
    * `limit` *(optional, number, default: 50)*: Number of rows.
    * `offset` *(optional, number, default: 0)*: Pagination offset.
  * **Response (`200 OK`)**:
    ```json
    {
      "total": 1248,
      "page": 1,
      "accounts": [
        {
          "id": "rp-092",
          "holder": "Rahul Sharma",
          "accountNumber": "0923 5678 9012",
          "customerId": "CUST-778899",
          "type": "Savings",
          "kyc": "Verified",
          "score": 64,
          "risk": "Medium",
          "balance": "₹6,84,220",
          "branch": "Connaught Place Branch"
        }
      ]
    }
    ```

* **`GET /api/accounts/:id`**
  * **Summary**: Get detailed account record by ID.
  * **Path Parameters**:
    * `id`: Account identifier (e.g. `rp-092`, `rp-138`).
  * **Response (`200 OK`)**:
    ```json
    {
      "id": "rp-092",
      "holder": "Rahul Sharma",
      "accountNumber": "0923 5678 9012",
      "customerId": "CUST-778899",
      "ifscCode": "RKSH0000108",
      "branchCode": "CP-0108",
      "type": "Savings",
      "kyc": "Verified",
      "score": 64,
      "risk": "Medium",
      "balance": "₹6,84,220",
      "branch": "Connaught Place Branch",
      "opened": "12 Feb 2021",
      "phone": "+91 98765 11442",
      "email": "rahul.sharma@northmail.in",
      "isHeld": false
    }
    ```

* **`POST /api/accounts/:id/hold`**
  * **Summary**: Place or release compliance hold on an account.
  * **Path Parameters**:
    * `id`: Account ID.
  * **Request Body**:
    ```json
    {
      "held": true,
      "reason": "Suspected mule account pattern"
    }
    ```
  * **Response (`200 OK`)**:
    ```json
    {
      "success": true,
      "accountId": "rp-092",
      "isHeld": true,
      "message": "Account placed on compliance hold"
    }
    ```

---

### 3. Transactions Endpoints

* **`GET /api/transactions`**
  * **Summary**: Get live transaction monitoring log.
  * **Query Parameters**:
    * `accountId` *(optional)*: Filter transactions for a specific account.
    * `status` *(optional)*: `Cleared`, `Review`, `Blocked`.
    * `type` *(optional)*: `Debit`, `Credit`.
    * `search` *(optional)*: Filter by customer or counterparty.
  * **Response (`200 OK`)**:
    ```json
    {
      "transactions": [
        {
          "id": "TXN-90214",
          "accountId": "rp-092",
          "customer": "Rahul Sharma",
          "counterparty": "Apex Trading Co",
          "channel": "IMPS",
          "amount": "₹1,85,000",
          "type": "Debit",
          "risk": "High",
          "status": "Review",
          "time": "10:14 AM"
        }
      ]
    }
    ```

* **`POST /api/transactions/:id/action`**
  * **Summary**: Approve, block, or release a flagged transaction.
  * **Request Body**:
    ```json
    {
      "action": "Block", // "Clear" | "Block" | "Escalate"
      "notes": "Unverified counterparty"
    }
    ```
  * **Response (`200 OK`)**:
    ```json
    {
      "id": "TXN-90214",
      "status": "Blocked",
      "updatedAt": "2026-08-25T13:20:00Z"
    }
    ```

---

### 4. Risk Alerts & Triage Endpoints

* **`GET /api/alerts`**
  * **Summary**: Get open and triage risk alerts.
  * **Query Parameters**:
    * `status` *(optional)*: `All`, `Open`, `Investigating`, `Resolved`.
  * **Response (`200 OK`)**:
    ```json
    {
      "alerts": [
        {
          "id": "ALT-408",
          "accountId": "rp-092",
          "account": "Rahul Sharma",
          "title": "Sudden volume surge",
          "detail": "3 high-value debits in 18 minutes to newly added payees.",
          "severity": "Critical",
          "status": "Open",
          "time": "11:28 AM"
        }
      ]
    }
    ```

* **`PATCH /api/alerts/:id`**
  * **Summary**: Update alert triage status.
  * **Request Body**:
    ```json
    {
      "status": "Investigating" // "Open" | "Investigating" | "Resolved"
    }
    ```
  * **Response (`200 OK`)**:
    ```json
    {
      "id": "ALT-408",
      "status": "Investigating",
      "updated": true
    }
    ```

---

## 6. Frontend & Client Integration

In the frontend React application (`artifacts/rakshapay-admin` or `frontend`), API calls use the custom fetch helper in [lib/api-client-react](file:///Users/pshreetam/Desktop/rakshapay-admin-main/lib/api-client-react):

```javascript
import { customFetch } from "@workspace/api-client-react";

// Fetch health status
const health = await customFetch("/healthz");

// Fetch account details
const account = await customFetch(`/accounts/${accountId}`);
```

---

## 7. ML Model & Risk Engine Integration

The Python machine learning pipeline in [ml/](file:///Users/pshreetam/Desktop/rakshapay-admin-main/ml) trains Student-Teacher anomaly detectors and Isolation Forests on unified transaction streams (`train_student.py`, `train_autoencoder.py`).

### Scoring Endpoint Format:
```http
POST /api/ml/score-transaction
Content-Type: application/json

{
  "amount": 185000.00,
  "account_age_days": 1240,
  "device_new": 1,
  "payee_new": 1,
  "channel": "IMPS",
  "hour_of_day": 10
}
```

**Response (`200 OK`)**:
```json
{
  "riskScore": 84,
  "riskLevel": "High",
  "anomalyConfidence": 0.912,
  "primarySignals": [
    "First-time high-value payee",
    "Rapid debit velocity"
  ],
  "recommendedAction": "Review"
}
```

---

## 8. CLI Commands Reference

```bash
# 1. Start Express Backend API Server
cd artifacts/api-server
PORT=3000 node src/index.js

# 2. Push Database Schema to PostgreSQL
cd lib/db
npx drizzle-kit push

# 3. Open Visual Drizzle Database Studio GUI
cd lib/db
npx drizzle-kit studio

# 4. Generate OpenAPI Zod Validations & Client Hooks
cd lib/api-spec
npm run generate

# 5. Start Admin Dashboard Frontend
cd artifacts/rakshapay-admin
npm run dev
```

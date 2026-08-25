# RakshaPay Monorepo JavaScript Architecture & Guidelines

## Overview
This repository has been fully converted from TypeScript to **pure modern JavaScript (ESM) and JSX**. All TypeScript files (`.ts`, `.tsx`, `tsconfig*.json`, `.tsbuildinfo`) have been removed and replaced with standard JavaScript tooling and `jsconfig.json` configurations.

---

## Workspace Structure & Packages

| Package | Path | Type | Key Files |
| :--- | :--- | :--- | :--- |
| **RakshaPay Admin** | `artifacts/rakshapay-admin` | React / Vite (JSX) | `src/App.jsx`, `src/main.jsx`, `vite.config.js`, `jsconfig.json` |
| **Backend API** | `artifacts/api-server` | Express (ESM JS) | `src/index.js`, `src/app.js`, `src/routes/health.js`, `build.mjs` |
| **Mockup Sandbox** | `artifacts/mockup-sandbox` | Vite / React (JSX) | `src/App.jsx`, `mockupPreviewPlugin.js`, `vite.config.js` |
| **User Frontend** | `frontend` | Vite / React (JSX) | `src/App.jsx`, `src/main.jsx`, `vite.config.js` |
| **Database Layer** | `lib/db` | Drizzle ORM (JS) | `src/index.js`, `src/schema/index.js`, `drizzle.config.js` |
| **API Validation** | `lib/api-zod` | Zod Schemas (JS) | `src/index.js`, `src/generated/api.js` |
| **React API Client** | `lib/api-client-react` | TanStack Query (JS) | `src/index.js`, `src/custom-fetch.js`, `src/generated/api.js` |
| **Scripts** | `scripts` | Node.js (ESM) | `src/hello.js` |

---

## Key Development Rules & Standards

### 1. File Extensions
* Use `.js` for all JavaScript logic, utilities, configs, and server files.
* Use `.jsx` for all React component files.
* **Do not create `.ts` or `.tsx` files.**

### 2. Path Aliases (`@/*`)
* In all frontends (`rakshapay-admin`, `mockup-sandbox`), use the `@/` alias to reference `src/`:
  ```javascript
  import { Button } from "@/components/ui/button.jsx";
  import { cn } from "@/lib/utils.js";
  ```
* These mappings are declared in [artifacts/rakshapay-admin/jsconfig.json](file:///Users/pshreetam/Desktop/rakshapay-admin-main/artifacts/rakshapay-admin/jsconfig.json) and [artifacts/mockup-sandbox/jsconfig.json](file:///Users/pshreetam/Desktop/rakshapay-admin-main/artifacts/mockup-sandbox/jsconfig.json).

### 3. Component Architecture
* React UI components use standard function syntax and `React.forwardRef`:
  ```jsx
  import * as React from "react";
  import { Slot } from "@radix-ui/react-slot";
  import { cn } from "@/lib/utils.js";

  const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn("btn", className)} ref={ref} {...props} />;
  });
  Button.displayName = "Button";
  export { Button };
  ```

### 4. Account State & Navigation Synchronization
* The admin app shares an `AccountContext` via `useAccount()` in [artifacts/rakshapay-admin/src/App.jsx](file:///Users/pshreetam/Desktop/rakshapay-admin-main/artifacts/rakshapay-admin/src/App.jsx).
* When a user clicks any account in the Overview, Alerts, or Transactions table:
  1. `activeAccountId` is updated in state and saved to `localStorage` (`rakshapay-active-account`).
  2. The sidebar **Accounts** item dynamically links to `/accounts/${activeAccountId}`.
  3. Visiting `/accounts` or clicking Accounts in the sidebar always displays the last selected account instead of a static default.

---

## Running Development Servers

### From the Root Directory:
```bash
# Start RakshaPay Admin Dashboard (http://localhost:5173)
pnpm run dev

# Start Backend API Server
pnpm run dev:api

# Start Mockup Sandbox
pnpm run dev:sandbox
```

### Direct Package Execution:
```bash
# Admin Frontend
cd artifacts/rakshapay-admin && pnpm run dev

# User Frontend
cd frontend && npm run dev

# API Server
cd artifacts/api-server && PORT=3000 node src/index.js
```

---

## Version Control Safety
* All changes must remain local until the user explicitly requests to push.
* Never execute `git push` without confirmation.

/**
 * Api specification
 * OpenAPI spec version: 0.1.0
 */
import * as zod from "zod";

/**
 * Returns server health status
 * @summary Health check
 */
export const HealthCheckResponse = zod.object({
  status: zod.string(),
});

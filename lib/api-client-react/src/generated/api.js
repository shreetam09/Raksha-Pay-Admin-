/**
 * Api specification
 * OpenAPI spec version: 0.1.0
 */
import { useQuery } from "@tanstack/react-query";
import { customFetch } from "../custom-fetch.js";

/**
 * Returns server health status
 * @summary Health check
 */
export const getHealthCheckUrl = () => {
  return `/api/healthz`;
};

export const healthCheck = async (options) => {
  return customFetch(getHealthCheckUrl(), {
    ...options,
    method: "GET",
  });
};

export const getHealthCheckQueryKey = () => {
  return [`/api/healthz`];
};

export const getHealthCheckQueryOptions = (options) => {
  const { query: queryOptions, request: requestOptions } = options ?? {};

  const queryKey = queryOptions?.queryKey ?? getHealthCheckQueryKey();

  const queryFn = ({ signal }) => healthCheck({ signal, ...requestOptions });

  return { queryKey, queryFn, ...queryOptions };
};

/**
 * @summary Health check
 */
export function useHealthCheck(options) {
  const queryOptions = getHealthCheckQueryOptions(options);

  const query = useQuery(queryOptions);

  return { ...query, queryKey: queryOptions.queryKey };
}

import { createFileRoute, redirect } from "@tanstack/react-router";
import { listDatasets } from "@/api";

export const Route = createFileRoute("/")({
  beforeLoad: async () => {
    const datasets = await listDatasets();
    if (datasets.length > 0) {
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      throw redirect({ to: "/$dataset", params: { dataset: datasets[0] } });
    }
  },
});

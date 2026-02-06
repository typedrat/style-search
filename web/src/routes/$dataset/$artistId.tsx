import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/$dataset/$artistId")({
  component: () => null, // Parent handles rendering
});

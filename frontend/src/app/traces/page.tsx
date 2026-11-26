import TraceExplorer from "@/features/traces/TraceExplorer";

export default async function TracesPage({ searchParams }: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const filters = Object.fromEntries(
    Object.entries(params).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
  );
  return <TraceExplorer key={JSON.stringify(filters)} initialFilters={filters} />;
}

import { Source } from "./SourceBubble";

export function InlineCitation(props: {
  source: Source;
  sourceNumber: number;
  highlighted: boolean;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
}) {
  const { source, sourceNumber, highlighted, onMouseEnter, onMouseLeave } =
      props;
  return (
      <a
          href={source.url}
          target="_blank"
          className={`relative bottom-1.5 text-xs border rounded px-1 py-0.5 
        ${highlighted ? "bg-[#f0f0f0] border-[#d0d0d0]" : "bg-[#e0e0e0] border-[#c0c0c0]"}`}
          onMouseEnter={onMouseEnter}
          onMouseLeave={onMouseLeave}
          style={{
            display: "inline-block",
            textAlign: "center",
            color: "#262629", /* Dark text color */
          }}
      >
        {sourceNumber}
      </a>
  );
}
import "react-toastify/dist/ReactToastify.css";
import { Card, CardBody, Heading } from "@chakra-ui/react";
import { sendFeedback } from "../utils/sendFeedback";

export type Source = {
    url: string;
    title: string;
};

export function SourceBubble({
                                 source,
                                 highlighted,
                                 onMouseEnter,
                                 onMouseLeave,
                                 runId,
                             }: {
    source: Source;
    highlighted: boolean;
    onMouseEnter: () => any;
    onMouseLeave: () => any;
    runId?: string;
}) {
    const displayTitle = source.title?.trim() ? source.title : "View Source";

    return (
        <Card
            onClick={async () => {
                window.open(source.url, "_blank");
                if (runId) {
                    await sendFeedback({
                        key: "user_click",
                        runId,
                        value: source.url,
                        isExplicit: false,
                    });
                }
            }}
            backgroundColor={highlighted ? "#f0f0f0" : "#e0e0e0"}
            onMouseEnter={onMouseEnter}
            onMouseLeave={onMouseLeave}
            cursor={"pointer"}
            alignSelf={"stretch"}
            height="auto"
            overflow={"hidden"}
            padding="8px 12px"
            display="flex"
            alignItems="center"
            justifyContent="center"
        >
            <CardBody padding="0">
                <Heading
                    size={"sm"}
                    fontWeight={"normal"}
                    color={"#262629"}
                    textAlign="center"
                    whiteSpace="nowrap"
                >
                    {displayTitle}
                </Heading>
            </CardBody>
        </Card>
    );
}
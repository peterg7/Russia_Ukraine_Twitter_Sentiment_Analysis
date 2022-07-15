
// Set the dimensions and margins of the graph
const SCREEN_DIMENSIONS = { 
    width: 740,
    height: 640,
    leftMargin: 75,
    rightMargin: 115,
    topMargin: 25,
    bottomMargin: 45
};
SCREEN_DIMENSIONS.innerWidth = SCREEN_DIMENSIONS.width - SCREEN_DIMENSIONS.leftMargin - SCREEN_DIMENSIONS.rightMargin;
SCREEN_DIMENSIONS.innerHeight = SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.topMargin - SCREEN_DIMENSIONS.bottomMargin;


// Run `cleanDataOutput` in `file_manager.py` to ensure most recent data
const DATA_LOC = '../../data/top_10_russia_ukraine_hashtags.csv';
// const DATA_LOC = 'https://raw.githubusercontent.com/peterg7/Russia_Ukraine_Twitter_Sentiment_Analysis/main/visualizations/part1/data/slava_tweets_sentiment_distrib.csv';
const TITLE = "10 Most Frequently Used Hashtags (Generalized)";

// Append the svg object to the body of the page
var svg = d3.select('body').append('svg')
            .attr('width', SCREEN_DIMENSIONS.width)
            .attr('height', SCREEN_DIMENSIONS.height)
            .attr('viewBox', [0, 0, SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height])
            .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

var canvas = svg.append('g')
    .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin},${SCREEN_DIMENSIONS.topMargin})`);

const xParams = {
    value: row => row.count,
    label: "Total Count",
    format: undefined,
    domain: undefined,
    range: [SCREEN_DIMENSIONS.leftMargin, SCREEN_DIMENSIONS.innerWidth],
    scale: d3.scaleLinear()
};
xParams.axis = d3.axisBottom(xParams.scale)
                    .ticks(SCREEN_DIMENSIONS.height / 60, xParams.format);

const yParams = {
    value: row => row.hashtag,
    label: 'Hashtag',
    padding: 0.15,
    domain: undefined,
    range: [SCREEN_DIMENSIONS.topMargin, SCREEN_DIMENSIONS.innerHeight],
    scale: d3.scaleBand()
};
yParams.axis = d3.axisLeft(yParams.scale)
                    .tickSizeOuter(0);

const colorParams = {
    value: d3.schemeTableau10
};

const zParams = {
    value: row => row.sentiment,
    label: 'Sentiment',
    padding: 0.1,
    domain: undefined,
    scale: d3.scaleOrdinal()
};


// var titleGroup = canvas.append("text")
//     .attr("class", "title")
//     .attr('transform', `translate(${(SCREEN_DIMENSIONS.width / 2)},${0 - (SCREEN_DIMENSIONS.topMargin)})`)
//     .text(TITLE);

var legendGroup = svg.append('g')
    .attr("class", "legendOrdinal")
    // .attr("text-anchor", "center")
    .attr("transform", `translate(${SCREEN_DIMENSIONS.innerWidth + SCREEN_DIMENSIONS.leftMargin + 30},${SCREEN_DIMENSIONS.topMargin * 2})`);


// xAxisGroup.append('text')
//     .attr('class', 'x-axis-label')
//     .attr('x', SCREEN_DIMENSIONS.innerWidth / 2)
//     .attr('y', 50)
//     .text(xParams.label);

// yAxisGroup.append('text')
//     .attr('class', 'y-axis-label')
//     .attr('x', -SCREEN_DIMENSIONS.topMargin * 0.75)
//     .attr('y', -SCREEN_DIMENSIONS.leftMargin * 1.5)
//     .text(yParams.label);


const preprocess = (row, i) => {
    // Know there are no null-values from preprocessing
    return {
        'hashtag': row['hashtag'],
        'sentiment': row['sentiment'],
        'count': parseInt(row['sentiment_count']),
        'frequency': parseFloat(row['freq'])
    }
};

d3.csv(DATA_LOC, preprocess).then((data, i) => {

    // console.log(data)

    // Grab metrics/value names (headers)
    const X = d3.map(data, xParams.value);
    const Y = d3.map(data, yParams.value);
    const Z = d3.map(data, zParams.value);

    // yParams.domain = d3.groupSort(data, D => d3.sum(D, d => -d.count), d => d.sentiment)
    colorParams.value = d3.schemeTableau10[Z.length]

    // console.log(X)
    // console.log(Y)
    // console.log(Z)

    // Compute default domains, and unique the x- and z-domains.
    if (xParams.domain === undefined) xParams.domain = [0, d3.max(X)];
    if (yParams.domain === undefined) yParams.domain = Y;
    if (zParams.domain === undefined) zParams.domain = Z;
    yParams.domain = new d3.InternSet(yParams.domain);
    zParams.domain = new d3.InternSet(zParams.domain);

    // console.log(xParams.domain)
    // console.log(yParams.domain)
    // console.log(zParams.domain)

    // Omit any data not present in both the x- and z-domain.
    const I = d3.range(Y.length).filter(i => yParams.domain.has(Y[i]) && zParams.domain.has(Z[i]));
    
    xParams.scale.domain(xParams.domain)
                    .range(xParams.range);
    yParams.scale.domain(yParams.domain)
                    .range(yParams.range)
                    .paddingInner(yParams.padding);
    zParams.scale.domain(zParams.domain)
    // .domain(['negative', 'positive', 'neutral'])
                .range(['#e41a1c', '#4daf4a', '#377eb8'])
    const yzScale = d3.scaleBand(zParams.domain, [0, yParams.scale.bandwidth()])
                        .padding(zParams.padding);

    svg.append('g')
        .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin},${SCREEN_DIMENSIONS.height-SCREEN_DIMENSIONS.bottomMargin})`)
        .call(xParams.axis)
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll(".tick line").clone()
            .attr("y2", -SCREEN_DIMENSIONS.innerHeight)
            .attr("stroke-opacity", 0.1))
        .call(g => g.append("text")
            .attr("class", "x-axis-label")
            .attr("x", SCREEN_DIMENSIONS.width/2 - SCREEN_DIMENSIONS.rightMargin/2)
            .attr("y", SCREEN_DIMENSIONS.bottomMargin)
            .attr("fill", "currentColor")
            .attr("text-anchor", "center")
            .text(xParams.label));

    const bar = canvas.append("g")
                        .selectAll("rect")
                        .data(I)
                        .join("rect")
                        .attr("x", SCREEN_DIMENSIONS.leftMargin) //i => xParams.scale(X[i]))
                        .attr("y", i => yParams.scale(Y[i]) + yzScale(Z[i]))
                        .attr("width", i => xParams.scale(X[i]) - xParams.scale(0))
                        .attr("height", yzScale.bandwidth())
                        .attr("fill", i => zParams.scale(Z[i]));

    canvas.append('text')
            .attr('class', 'title')
            .attr("text-anchor", "center")
            .attr('x', SCREEN_DIMENSIONS.innerWidth / 2)
            .attr('y', 0 - SCREEN_DIMENSIONS.topMargin / 2)
            .text(TITLE);

    canvas.append('g')
            .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin},0)`)
            .call(yParams.axis)
            .call(g => g.append("text")
                // .attr("class", "y-axis-label")
                .attr("transform", `translate(-${SCREEN_DIMENSIONS.leftMargin*1.75}, ${SCREEN_DIMENSIONS.innerHeight/2})rotate(-90)`)
                .attr("fill", "#635F5D")
                .attr("text-anchor", "center")
                .attr('font-size', "14pt")
                .text(yParams.label));

    // Define and build legend
    var legend = d3.legendColor()
        .shapeHeight(15)
        .orient("vertical")
        .scale(zParams.scale) 
        .title("Sentiment")
        .titleWidth(30);

    legendGroup.call(legend);
});

var raw_states_data = null;

function readTextFile(file) {
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = () => {
        if(rawFile.readyState === 4) {
            if(rawFile.status === 200 || rawFile.status == 0) {
                raw_states_data = rawFile.responseText;
            }
        }
    }
    rawFile.send(null);
};

function csvToArray(str, delimiter = ",") {
    const headers = str.slice(0, str.indexOf("\n")).split(delimiter);
    const rows = str.slice(str.indexOf("\n") + 1).split("\n");
    
    const arr = rows.map(row => {
        const values = row.split(delimiter);
        const el = headers.reduce((object, header, index) => {
            object[header] = values[index];
            return object;
        }, {});
        return el;
    });
    return arr;
};

readTextFile("us-population-state-age.csv");

var states = csvToArray(raw_states_data);
var ages = Object.keys(states[0]).slice(1);
var stateAges = ages.flatMap(age => states.map(d => ({state: d.name, age, population: d[age]})));

// ---------------------------------------------------------------- //

// Set the dimensions and margins of the graph
const SCREEN_DIMENSIONS = { 
    width: 640,
    height: 500,
    leftMargin: 40,
    rightMargin: 0,
    topMargin: 30,
    bottomMargin: 30
};
SCREEN_DIMENSIONS.innerWidth = SCREEN_DIMENSIONS.width - SCREEN_DIMENSIONS.leftMargin - SCREEN_DIMENSIONS.rightMargin;
SCREEN_DIMENSIONS.innerHeight = SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.topMargin - SCREEN_DIMENSIONS.bottomMargin;

const TITLE = "TEST";
const DATA_LOC = "us_state_ages.csv";

// Append the svg object to the body of the page
const svg = d3.select('body').append('svg')
            .attr('width', SCREEN_DIMENSIONS.width)
            .attr('height', SCREEN_DIMENSIONS.height)
            // .attr("viewBox", [-SCREEN_DIMENSIONS.leftMargin, -SCREEN_DIMENSIONS.topMargin, 
            //                     SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height])
            .attr('viewBox', [0, 0, SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height])
            .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

// const canvas = svg.append('g')
//     .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin},${SCREEN_DIMENSIONS.topMargin})`);
        

// var xAxisGroup = canvas.append('g')
//     .attr("class", "x-axis")
//     .attr('transform', `translate(0, ${SCREEN_DIMENSIONS.innerHeight})`);

// var yAxisGroup = canvas.append('g')
//     .attr("class", "y axis");

// var titleGroup = svg.append("text")
//     .attr("class", "title")
//     .attr('transform', `translate(${(SCREEN_DIMENSIONS.width / 2)},${0 - (SCREEN_DIMENSIONS.topMargin)})`)
//     .text(TITLE);

// var legendGroup = svg.append('g')
//     .attr("class", "legendOrdinal")
//     .attr("transform", `translate(${SCREEN_DIMENSIONS.innerWidth - SCREEN_DIMENSIONS.rightMargin},${SCREEN_DIMENSIONS.topMargin * 2})`);
        

const xParams = {
    value: row => row.state,
    label: 'Total Count',
    padding: 0.1,
    domain: d3.groupSort(stateAges, D => d3.sum(D, d => -d.population), d => d.state).slice(0, 6),
    range: [SCREEN_DIMENSIONS.leftMargin, SCREEN_DIMENSIONS.width - SCREEN_DIMENSIONS.rightMargin],
    scale: d3.scaleBand()
};
xParams.axis = d3.axisBottom(xParams.scale)
                    .tickSizeOuter(0);

const yParams = {
    value: row => row.population / 1e6,
    label: "↑ Population (millions)",
    format: undefined,
    domain: undefined,
    range: [SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.bottomMargin, SCREEN_DIMENSIONS.topMargin],
    scale: d3.scaleLinear()
};
yParams.axis = d3.axisLeft(yParams.scale)
                    .ticks(SCREEN_DIMENSIONS.height / 60, yParams.format);

const colorParams = {
    value: d3.schemeSpectral[ages.length]
};

const zParams = {
    value: row => row.age,
    label: 'Hashtag',
    padding: 0.05,
    domain: ages,
    scale: d3.scaleOrdinal()
};


const preprocess = (row, i) => {
    // Know there are no null-values from external preprocessing
    return {
        state: row['state'],
        age: row['age'],
        population: parseInt(row['population'])
    }
};


d3.csv(DATA_LOC, preprocess).then((data, i) => {

    const X = d3.map(data, xParams.value);
    const Y = d3.map(data, yParams.value);
    const Z = d3.map(data, zParams.value);
    
    // Compute default domains, and unique the x- and z-domains.
    if (xParams.domain === undefined) xParams.domain = X;
    if (yParams.domain === undefined) yParams.domain = [0, d3.max(Y)];
    if (zParams.domain === undefined) zParams.domain = Z;
    xParams.domain = new d3.InternSet(xParams.domain);
    zParams.domain = new d3.InternSet(zParams.domain);

    console.log(xParams.domain)
    
    // Omit any data not present in both the x- and z-domain.
    const I = d3.range(X.length).filter(i => xParams.domain.has(X[i]) && zParams.domain.has(Z[i]));
    
    xParams.scale.domain(xParams.domain)
                    .range(xParams.range)
                    .paddingInner(xParams.padding);
    yParams.scale.domain(yParams.domain)
                    .range(yParams.range);
    zParams.scale.domain(zParams.domain)
                    .range(colorParams.value);
    const xzScale = d3.scaleBand(zParams.domain, [0, xParams.scale.bandwidth()])
                        .padding(zParams.padding);
    
    // Compute titles.
    if (TITLE === undefined) {
        const formatValue = yParams.scale.tickFormat(100, yParams.format);
        title = i => `${X[i]}\n${Z[i]}\n${formatValue(Y[i])}`;
    } else {
        const O = d3.map(data, d => d);
        const T = TITLE;
        title = i => T(O[i], i, data);
    }
    
    svg.append('g')
        .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin},0)`)
        .call(yParams.axis)
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll(".tick line").clone()
        .attr("x2", SCREEN_DIMENSIONS.innerWidth)
        .attr("stroke-opacity", 0.1))
        .call(g => g.append("text")
        .attr("x", -SCREEN_DIMENSIONS.leftMargin)
        .attr("y", 10)
        .attr("fill", "currentColor")
        .attr("text-anchor", "start")
        .text(yParams.label));
    
    const bar = svg.append("g")
                    .selectAll("rect")
                    .data(I)
                    .join("rect")
                    .attr("x", i => xParams.scale(X[i]) + xzScale(Z[i]))
                    .attr("y", i => yParams.scale(Y[i]))
                    .attr("width", xzScale.bandwidth())
                    .attr("height", i => yParams.scale(0) - yParams.scale(Y[i]))
                    .attr("fill", i => zParams.scale(Z[i]));
    
    if (TITLE) bar.append("title")
                    .text(TITLE);
    
    svg.append('g')
        .attr('transform', `translate(0,${SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.bottomMargin})`)
        .call(xParams.axis);
});


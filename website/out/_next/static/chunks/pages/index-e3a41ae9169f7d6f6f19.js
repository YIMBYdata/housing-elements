(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{2562:function(e,t,s){"use strict";s.r(t),s.d(t,{useFetch:function(){return x},useBufferSelector:function(){return j},default:function(){return _}});var n=s(5893),i=s(6265),r=s(7294),a=s(4555),l=s(862),o=s(9008),c=s(7905),d=(s(801),s(8767));function u(e,t){var s;if("undefined"===typeof Symbol||null==e[Symbol.iterator]){if(Array.isArray(e)||(s=function(e,t){if(!e)return;if("string"===typeof e)return m(e,t);var s=Object.prototype.toString.call(e).slice(8,-1);"Object"===s&&e.constructor&&(s=e.constructor.name);if("Map"===s||"Set"===s)return Array.from(e);if("Arguments"===s||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(s))return m(e,t)}(e))||t&&e&&"number"===typeof e.length){s&&(e=s);var n=0,i=function(){};return{s:i,n:function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}},e:function(e){throw e},f:i}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}var r,a=!0,l=!1;return{s:function(){s=e[Symbol.iterator]()},n:function(){var e=s.next();return a=e.done,e},e:function(e){l=!0,r=e},f:function(){try{a||null==s.return||s.return()}finally{if(l)throw r}}}}function m(e,t){(null==t||t>e.length)&&(t=e.length);for(var s=0,n=new Array(t);s<t;s++)n[s]=e[s];return n}function h(e,t){var s=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),s.push.apply(s,n)}return s}function p(e){for(var t=1;t<arguments.length;t++){var s=null!=arguments[t]?arguments[t]:{};t%2?h(Object(s),!0).forEach((function(t){(0,i.Z)(e,t,s[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(s)):h(Object(s)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(s,t))}))}return e}function x(e){return(0,d.useQuery)(e,(function(){return fetch(e).then((function(e){return e.json()}))}),{staleTime:1/0})}var f=(0,c.ZP)({accessToken:"pk.eyJ1Ijoic2lkLWthcCIsImEiOiJjamRpNzU2ZTMxNWE0MzJtZjAxbnphMW5mIn0.b6m4jgFhPOPOYOoaNGmogQ"}),y=[[-123.1360271,36.97640742],[-121.53590925,38.82979915]];function j(){var e=(0,r.useState)(25),t=e[0],s=e[1],i=[0,5,10,25,50,75,100].map((function(e){return(0,n.jsxs)("span",{children:[(0,n.jsx)("input",{id:"buffer-"+e,type:"radio",checked:t==e,value:"conservative",onChange:function(){return s(e)}}),(0,n.jsxs)("label",{htmlFor:"buffer-"+e,className:"ml-1 mr-3 text-sm",children:[e," feet"]})]},e)}));return{buffer:t,bufferInput:(0,n.jsxs)("div",{children:[(0,n.jsx)("p",{className:"text-sm text-gray-500",children:"Geocoding buffer size:"}),(0,n.jsx)("div",{className:"max-w-xs",children:i})]})}}var v={keys:["name"],threshold:.1,distance:5};function g(e,t,s,i){return(0,n.jsx)("button",p(p({className:i},e),{},{children:t.name}))}var b={"line-color":"black","line-width":.5},N=[{name:"All sites",value:"overall"},{name:"Nonvacant sites",value:"nonvacant"},{name:"Vacant sites",value:"vacant"}],w=[{name:"APN only",value:"apn_only"},{name:"APN and geocoding",value:"apn_and_geo"}];function _(){var e=(0,r.useState)("Overview"),t=e[0],s=e[1],i="Overview"==t,l=(0,r.useState)(null),d=l[0],m=l[1],h=i?null:"data/"+t+"/sites_with_matches.geojson",_=i?null:"data/"+t+"/permits.geojson",O=x("data/summary.json").data,P=(0,r.useMemo)((function(){return function(e){var t=(e||[]).slice();t.sort((function(e,t){return e.city.localeCompare(t.city)}));for(var s={},n=[{name:"Overview",value:"Overview"}],i=0;i<t.length;i++){var r=t[i];n.push(p(p({},r),{},{value:r.city,name:r.city})),s[r.city]=i+1}return[n,s]}(O)}),[O]),k=P[0],A=P[1],L=(0,r.useMemo)((function(){return k[A[t]]}),[O,t]),I=null,E=(0,r.useState)("overall"),F=E[0],T=E[1],W=(0,r.useState)("apn_and_geo"),D=W[0],H=W[1],J=j(),Z=J.buffer,z=J.bufferInput,U="apn_only"==D?"results_apn_only":"results_apn_and_geo_".concat(Z,"ft"),B="geo_matched_".concat(Z,"ft"),G={"fill-color":["case",["all",["get","apn_matched"],["get",B]],"green",["get","apn_matched"],"yellow",["get",B],"blue","red"],"fill-opacity":.3},Q=["get","P(dev)",["get",F,["get",U]]],q={"fill-color":["interpolate",["linear"],Q,-.1,"white",.5,"blue",1,"blue"],"fill-opacity":i?.5:0};return I=(0,n.jsxs)("div",{className:"mx-auto mb-10 align-center items-center justify-center flex flex-col",children:[(0,n.jsxs)("div",{children:[(0,n.jsx)("h1",{className:"text-5xl m-2 text-center",children:"Development on Housing Element Sites"}),(0,n.jsx)("h2",{className:"text-xl mb-4 text-center",children:"Sidharth Kapur, Salim Damderji, Christopher S. Elmendorf, Paavo Monkkonen"}),(0,n.jsxs)("div",{className:"max-w-3xl mx-2 mt-6 mb-10 leading-snug",children:[(0,n.jsxs)("p",{children:["This map is a companion to the Lewis Center report, ",(0,n.jsx)("a",{className:"text-blue-500 hover:underline",href:"#",children:"\"What Gets Built on Sites that Cities 'Make Available' for Housing?\""}),"."]}),(0,n.jsx)("p",{className:"mt-4",children:"Below, the overview shows our likelihood of development estimates\u2014the probability that housing was built on a site listed in a jurisdiction's 5th cycle Housing Element\u2014for Bay Area cities. It allows you to visualize likelihoods for vacant and nonvacant sites, as well as under different matching criteria. Because building permits and sites do not always match using parcel numbers, we use both assessor parcel numbers and a spatial overlay approach with different buffer sizes. See the full report for more detail."}),(0,n.jsx)("p",{className:"mt-4",children:'The views for individual cities (accessible using the dropdown at the top) allows comparing the location of housing inventory sites for a city to the locations that received new housing building permits in 2015-2019. The "buffer size" control allows you to compare which sites are considered "matched" (indicated by the color of the housing inventory site) to a permit as the geocoding buffer size varies.'})]})]}),(0,n.jsxs)("div",{className:"lg:grid lg:grid-cols-3 flex flex-col",children:[(0,n.jsx)("div",{className:"m-4 col-span-1",children:(0,n.jsx)(a.ZP,{search:!0,onChange:s,options:k,fuseOptions:v,value:t,renderOption:g})}),(0,n.jsx)("div",{className:"col-span-1",children:(0,n.jsx)("h1",{className:"mt-4 text-center text-4xl",children:t})}),(0,n.jsx)("div",{className:"col-span-1 m-4",children:!i&&z})]}),(0,n.jsx)("div",{className:"w-full justify-center flex flex-row",children:(0,n.jsxs)(f,{style:"mapbox://styles/mapbox/streets-v9",containerStyle:{height:"700px",width:"1000px"},fitBounds:(null===L||void 0===L?void 0:L.bounds)||y,onMouseMove:function(e,t){var s=e.queryRenderedFeatures(t.point).filter((function(e){return"sitesWithMatches"==e.source||"permits"==e.source})).length>0;e.getCanvas().style.cursor=s?"pointer":"default"},onClick:function(e,t){var s=e.queryRenderedFeatures(t.point,{layers:["sitesWithMatchesLayer","permitsLayer","summaryLayer"]});s.length>0?m({layer:s[0].layer.id,element:s[0],location:t.lngLat.toArray()}):m(null)},children:[(0,n.jsx)(c.X$.Consumer,{children:function(e){var t,s=u(e.getStyle().layers);try{for(s.s();!(t=s.n()).done;){var r=t.value;r.id.includes("place-")&&e.setLayoutProperty(r.id,"visibility",i?"none":"visible")}}catch(a){s.e(a)}finally{s.f()}return(0,n.jsx)(n.Fragment,{})}}),(0,n.jsx)(c.Hw,{id:"permits",geoJsonSource:{data:_,type:"geojson"}}),(0,n.jsx)(c.Hw,{id:"sitesWithMatches",geoJsonSource:{data:h,type:"geojson"}}),(0,n.jsx)(c.Hw,{id:"summary",geoJsonSource:{data:"data/summary.geojson",type:"geojson"}}),(0,n.jsx)(c.Hw,{id:"summaryCentroids",geoJsonSource:{data:"data/summary_centroids.geojson",type:"geojson"}}),(0,n.jsx)(c.mh,{id:"sitesWithMatchesLayer",type:"fill",sourceId:"sitesWithMatches",paint:G,layout:{visibility:i?"none":"visible"}}),(0,n.jsx)(c.mh,{id:"sitesWithMatchesOutlineLayer",type:"line",sourceId:"sitesWithMatches",paint:b,minZoom:15,layout:{visibility:i?"none":"visible"}}),(0,n.jsx)(c.mh,{id:"sitesWithMatchesTextLayer",type:"symbol",sourceId:"sitesWithMatches",layout:{"text-field":"{site_capacity_units}",visibility:i?"none":"visible"},paint:{"text-color":"hsl(0, 0, 35%)"},minZoom:17}),(0,n.jsx)(c.mh,{id:"permitsLayer",type:"circle",sourceId:"permits",paint:{"circle-color":["case",["==",["get","permit_category"],"ADU"],"hsl(169, 76%, 50%)",["==",["get","permit_category"],"SU"],"hsl(169, 76%, 50%)","green"],"circle-radius":{base:1.75,stops:[[12,1.5],[15,4],[22,180]]}},layout:{visibility:i?"none":"visible"}}),(0,n.jsx)(c.mh,{id:"summaryLayer",type:"fill",sourceId:"summary",paint:q,layout:{visibility:i?"visible":"none"}}),(0,n.jsx)(c.mh,{id:"summaryTextLayer",type:"symbol",sourceId:"summaryCentroids",layout:{"text-field":["concat",["get","city"],"\n",["number-format",Q,{"max-fraction-digits":3}]],visibility:i?"visible":"none"},paint:{"text-color":"hsl(0, 0, 35%)"}}),d&&(0,n.jsx)(c.GI,{coordinates:d.location,children:C(Z,d.layer,d.element.properties)},d.element.id),i?(0,n.jsx)(n.Fragment,{}):S]})}),L&&M(L,Z),i&&(0,n.jsx)(n.Fragment,{children:(0,n.jsxs)("div",{className:"lg:grid lg:grid-cols-3 flex flex-col mb-20",children:[(0,n.jsxs)("div",{className:"m-4 col-span-1",children:[(0,n.jsx)("p",{className:"text-sm text-gray-500 mb-2",children:"Sites:"}),(0,n.jsx)(a.ZP,{onChange:T,options:N,fuseOptions:v,value:F,renderOption:g})]}),(0,n.jsxs)("div",{className:"m-4 col-span-1",children:[(0,n.jsx)("p",{className:"text-sm text-gray-500 mb-2",children:"Matching logic:"}),(0,n.jsx)(a.ZP,{onChange:H,options:w,fuseOptions:v,value:D,renderOption:g}),(0,n.jsxs)("div",{className:"text-sm text-gray-500 mt-2",children:[(0,n.jsxs)("p",{children:["(",(0,n.jsx)("span",{className:"font-bold",children:"APN"})," matches sites to permits using the county's assessor parcel number (APN), which uniquely identifies parcels."]}),(0,n.jsxs)("p",{children:[(0,n.jsx)("span",{className:"font-bold",children:"APN and geocoding"})," is more lenient: it matches a site to a permit if the APN matches, or if the site is within ",(0,n.jsx)("span",{className:"italic",children:"x"})," feet of a building permit.)"]})]})]}),(0,n.jsxs)("div",{className:"m-4 col-span-1",children:[z,(0,n.jsxs)("p",{className:"text-sm text-gray-500 mt-2",children:["(How lenient to be when geomatching: a site is considered matched if it is within ",(0,n.jsx)("span",{className:"italic",children:"x"})," feet of a building permit.)"]})]})]})})]}),(0,n.jsxs)(n.Fragment,{children:[(0,n.jsxs)(o.default,{children:[(0,n.jsx)("title",{children:"Development on Housing Element Sites - Map"}),(0,n.jsx)("meta",{name:"viewport",content:"width=device-width, initial-scale=1.0"})]}),(0,n.jsx)("body",{className:"bg-gray-50 font-sans",children:I})]})}var O="mx-1 w-3 h-3 inline-block border border-black border-opacity-100",P="mx-1 w-3 h-3 inline-block rounded-full",S=(0,n.jsxs)("div",{className:"p-2 absolute right-6 bottom-10 border bg-opacity-80 bg-white",children:[(0,n.jsxs)("div",{children:[(0,n.jsx)("span",{className:O,style:{backgroundColor:"red",opacity:.3}}),"Unmatched site"]}),(0,n.jsxs)("div",{children:[(0,n.jsx)("span",{className:O,style:{backgroundColor:"green",opacity:.3}}),"Matched site (APN and geocoding)"]}),(0,n.jsxs)("div",{children:[(0,n.jsx)("span",{className:O,style:{backgroundColor:"blue",opacity:.3}}),"Matched site (Geocoding only)"]}),(0,n.jsxs)("div",{children:[(0,n.jsx)("span",{className:O,style:{backgroundColor:"yellow",opacity:.3}}),"Matched site (APN only)"]}),(0,n.jsxs)("div",{children:[(0,n.jsx)("span",{className:P,style:{backgroundColor:"green"}}),"Permit (single-family or multifamily)"]}),(0,n.jsxs)("div",{children:[(0,n.jsx)("span",{className:P,style:{backgroundColor:"hsl(169, 76%, 50%)"}}),"Permit (ADU)"]})]});function k(e){return["overall","nonvacant","vacant"].map((function(t){var s,i=null==e[t]["P(dev)"]?(0,n.jsx)(n.Fragment,{children:"N/A"}):(0,n.jsxs)(n.Fragment,{children:[null===(s=e[t]["P(dev)"])||void 0===s?void 0:s.toFixed(3),"\xa0",(0,n.jsxs)("span",{className:"text-gray-400",children:["(",e[t]["# matches"].replaceAll(" ",""),")"]})]});return(0,n.jsx)("td",{className:"text-center",children:i})}))}function M(e,t){var s=e.results_apn_only,i=e["results_apn_and_geo_".concat(t,"ft")];return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsxs)("table",{className:"table-auto match-table mt-4",children:[(0,n.jsx)("thead",{children:(0,n.jsx)("tr",{children:(0,n.jsxs)("th",{className:"text-center",colSpan:4,children:["Likelihood of development for inventory sites in ",e.city,(0,n.jsx)("span",{className:"text-gray-500",children:"*"})]})})}),(0,n.jsxs)("tbody",{children:[(0,n.jsxs)("tr",{className:"bg-blue-300",children:[(0,n.jsx)("th",{children:"Matching logic"}),(0,n.jsx)("th",{className:"text-center",children:"Overall"}),(0,n.jsx)("th",{className:"text-center",children:"Nonvacant sites"}),(0,n.jsx)("th",{className:"text-center",children:"Vacant sites"})]}),(0,n.jsxs)("tr",{className:"bg-blue-100",children:[(0,n.jsx)("th",{className:"text-left",children:"APN"}),k(s)]}),(0,n.jsxs)("tr",{className:"bg-blue-100",children:[(0,n.jsx)("th",{className:"text-left",children:"APN and geocoding"}),k(i)]})]})]}),(0,n.jsx)("div",{className:"mt-4 text-sm text-gray-500 max-w-md",children:"(*The likelihood of development is extrapolated from the 2015-2019 period to 8 years by multiplying by 8/5.)"})]})}function A(e){var t,s=null===(t=e.permit_address)||void 0===t?void 0:t.toLowerCase();return s=s?(0,l.Q)(s):"[no address]",(0,n.jsx)("li",{children:(0,n.jsxs)("div",{children:[(0,n.jsxs)("p",{children:[s,": ",e.permit_units," units (",e.permit_year,")."]}),(0,n.jsxs)("p",{children:[(0,n.jsx)("span",{className:"font-bold",children:"Type:"})," ",e.permit_category]}),(0,n.jsxs)("p",{children:[(0,n.jsx)("span",{className:"font-bold",children:"Match type:"})," ",e.match_type]})]})})}function C(e,t,s){if("sitesWithMatchesLayer"==t){var i=JSON.parse(s["match_results_"+e+"ft"]);return(0,n.jsxs)("div",{className:"max-h-72 overflow-y-auto",children:[(0,n.jsxs)("p",{children:["Expected capacity in housing element: ",s.site_capacity_units," units"]}),i?(0,n.jsxs)(n.Fragment,{children:[(0,n.jsxs)("p",{children:["Matched permits (total ",i.map((function(e){return e.permit_units||0})).reduce((function(e,t){return e+t}),0)," units):"]}),(0,n.jsx)("ul",{className:"list-disc list-outside pl-5",children:i.map(A)})]}):(0,n.jsx)("p",{children:"Matched permits: None"})]})}if("permitsLayer"==t)return d=s,(0,n.jsxs)("div",{children:[(0,n.jsxs)("p",{children:[(0,l.Q)(d.permit_address.toLowerCase()),": ",d.permit_units," units (",d.permit_year,")."]}),(0,n.jsxs)("p",{children:[(0,n.jsx)("span",{className:"font-bold",children:"Type:"})," ",d.permit_category]})]});if("summaryLayer"==t){for(var r=p({},s),a=0,o=Object.keys(r);a<o.length;a++){var c=o[a];c.includes("results")&&(r[c]=JSON.parse(r[c]))}return M(r,e)}throw"Unknown layer clicked: "+t;var d}},5301:function(e,t,s){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return s(2562)}])}},function(e){e.O(0,[774,634,926,888,179],(function(){return t=5301,e(e.s=t);var t}));var t=e.O();_N_E=t}]);
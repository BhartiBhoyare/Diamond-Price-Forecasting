//variables
var shape = "CUSHION"
var weight = 0.43
var length = 4.38
var width = 4.34
var depth = 2.72
var colour = "M"
var clarity = "SI1"
var cut = "EX"
var polish = "EX"
var symmetry = "VG"
var fluorescence = "M"

const data = ["A cushion diamond is a square or rectangular-shaped diamond with rounded corners, resembling a pillow, known for its vintage appeal and brilliance.",
  "A heart diamond is a gemstone cut in the shape of a heart, symbolizing love and affection, commonly used in romantic jewelry such as engagement rings.",
  "A marquise diamond is a distinctive diamond cut characterized by an elongated shape with pointed ends, resembling a boat or an eye. Its design maximizes carat weight and creates the illusion of a larger diamond due to its elongated silhouette.",
  "An oval diamond is a modified brilliant-cut diamond with an elongated, elliptical shape, offering brilliance similar to round diamonds. Its elongated silhouette creates the illusion of lengthening the wearer's finger.",
  "An emerald diamond is a rectangular or square diamond cut with step-like facets, emphasizing clarity over brilliance. Its elongated shape and clean lines make it popular for both engagement rings and sophisticated jewelry designs.",
  "A round diamond is a classic diamond cut with a circular shape, maximizing brilliance and fire due to its symmetrical facets. It's the most popular choice for engagement rings and other fine jewelry settings.",
  "A pear diamond, also known as a teardrop shape, combines the characteristics of a round and marquise cut, featuring a rounded end tapering to a point. It offers elegance and versatility, commonly used in engagement rings and pendants.",
  "A princess diamond is a square or rectangular cut with pointed corners, known for its exceptional brilliance and modern appeal. It's a popular choice for engagement rings and other jewelry styles, offering a contemporary yet timeless look.",
  "A radiant diamond is a rectangular or square-shaped cut with trimmed corners and a brilliant facet pattern, blending the elegance of emerald and sparkle of round cuts. Its versatility and dazzling brilliance make it a favored choice for engagement rings and other jewelry pieces."
]

async function search() {
  window.scrollTo(0, 0)
  const price = document.getElementById("show");
  const loading = document.getElementById("loading");
  price.style.display = "none"
  loading.style.display = "block"
  const rawResponse = await fetch('https://diamond-api-112u.onrender.com/predict', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      "Shape": shape,
      "Weight": weight,
      "length": length,
      "width": width,
      "depth": depth,
      "Clarity": "SI1",
      "Colour": "M",
      "Cut": "EX",
      "Polish": "EX",
      "Symmetry": "VG",
      "Fluorescence": "M"
    })
  });
  await new Promise(resolve => setTimeout(resolve, 5000))
  const content = await rawResponse.json();


  const url = `https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert?from=USD&to=INR&amount=${content}`;
  const options = {
    method: 'GET',
    headers: {
      'X-RapidAPI-Key': 'e7563d52d0mshe3273d7bc376a98p14f4fcjsnef49df54a3de',
      'X-RapidAPI-Host': 'currency-conversion-and-exchange-rates.p.rapidapi.com'
    }
  };

  try {
    const response = await fetch(url, options);
    const result = await response.json();

    let inr = Math.round(result.result)

    price.textContent = '₹' + inr;
  } catch (error) {
    console.error(error);
  }

  price.style.display = "flex"
  loading.style.display = "none"

  document.getElementById("shape_img").src = `Images/Shapes/${shape}.jpg`

  var shapeno;

  if (shape == "CUSHION") {
    shapeno = 0;
  }
  else if (shape == "HEART") {
    shapeno = 1;
  }
  else if (shape == "MARQUISE") {
    shapeno = 2;
  }
  else if (shape == "OVAL") {
    shapeno = 3;
  }
  else if (shape == "EMERALD") {
    shapeno = 4;
  }
  else if (shape == "ROUND") {
    shapeno = 5;
  }
  else if (shape == "PEAR") {
    shapeno = 6;
  }
  else if (shape == "PRINCESS") {
    shapeno = 7;
  }
  else if (shape == "RADIANT") {
    shapeno = 8;
  }


  document.getElementById("carat").textContent = weight
  document.getElementById("color").textContent = colour
  document.getElementById("cut").textContent = cut
  document.getElementById("clarity").textContent = clarity

  document.getElementById("dia_des").textContent = data[shapeno]
}

window.addEventListener("scroll", (event) => {
  let scroll = this.scrollY;
  if (scroll == 0) {
    document.getElementById("navbar").className = "navbar"
  }
  else {
    document.getElementById("navbar").className = "navbar_shadow"
  }

});
scroll()

function change(name, src) {
  document.getElementById("heading").textContent = name;
  document.getElementById("heading_img").src = src;
}

function Shapef(Shape) {
  shapebtn()
  document.getElementById(Shape).style.backgroundColor = "#2e35f2"
  document.getElementById(Shape).style.color = "white"
  shape = Shape
}

function Weightf(Weight) {
  weight = Weight
}

function Lengthf(Length) {
  length = Length
}

function Widthf(Width) {
  width = Width
}

function Depthf(Depth) {
  depth = Depth
}

function Cutf(Cut, id) {
  cutbtn()
  document.getElementById(id).style.backgroundColor = "#2e35f2"
  document.getElementById(id).style.color = "white"
  cut = Cut
}

function Colourf(Colour, id) {
  colourbtn()
  document.getElementById(id).style.backgroundColor = "#2e35f2"
  document.getElementById(id).style.color = "white"
  colour = Colour
}

function Clarityf(Clarity, id) {
  claritybtn()
  document.getElementById(id).style.backgroundColor = "#2e35f2"
  document.getElementById(id).style.color = "white"
  clarity = Clarity
}

function Polishf(Polish, id) {
  polishbtn()
  document.getElementById(id).style.backgroundColor = "#2e35f2"
  document.getElementById(id).style.color = "white"
  polish = Polish
}

function Fluorescencef(Fluorescence, id) {
  fluorescencebtn()
  document.getElementById(id).style.backgroundColor = "#2e35f2"
  document.getElementById(id).style.color = "white"
  fluorescence = Fluorescence
}

function Symmetryf(Symmetry, id) {
  symmetrybtn()
  document.getElementById(id).style.backgroundColor = "#2e35f2"
  document.getElementById(id).style.color = "white"
  symmetry = Symmetry
}

function shapebtn() {
  document.querySelectorAll(".shapebtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}

function colourbtn() {
  document.querySelectorAll(".colourbtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}

function claritybtn() {
  document.querySelectorAll(".claritybtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}

function cutbtn() {
  document.querySelectorAll(".cutbtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}

function polishbtn() {
  document.querySelectorAll(".polishbtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}

function fluorescencebtn() {
  document.querySelectorAll(".fluorescencebtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}

function symmetrybtn() {
  document.querySelectorAll(".symmetrybtn").forEach((btn) => {
    btn.style.backgroundColor = "#e7e8f8"
    btn.style.color = "black"
  })
}


function rangefnc(range_value, div_value) {
  let value = document.getElementById(range_value).value
  document.getElementById(div_value).textContent = value

  if (div_value == "weight") {
    Weightf(value)
  }
  else if (div_value == "length") {
    Lengthf(value)
  }
  else if (div_value == "width") {
    Widthf(value)
  }
  else {
    Depthf(value)
  }
}


function advance() {
  var advance = document.getElementById("advance");
  var btn = document.getElementById("advancebtn");
  var arrow = document.getElementById("arrow");

  if (getComputedStyle(advance).display == 'none') {
    advance.style.display = "block";
    btn.textContent = "FEWER OPTION";
    arrow.src = "./Images/up arrow.png"
  }
  else {
    advance.style.display = "none";
    btn.textContent = "More Option"
    arrow.src = "./Images/down-arrow.png"
  }
}

function check() {
  if (localStorage.getItem("logged")) {
    window.location.href = "calculator.html";
  }
  else {
    window.location.href = "login.html"
  }
}

function run() {
  if (localStorage.getItem("logged")) {
    document.getElementById("welcome").innerHTML = `Welcome, <span id="user_name">${localStorage.getItem("user")}</span>`
    document.getElementById("login").textContent = "Log out"
  }
}
run()

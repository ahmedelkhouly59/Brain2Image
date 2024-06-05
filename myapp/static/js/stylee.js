const sr=ScrollReveal({
    origin:'top',
    distance:'30px',
    duration:2500,
    delay:400,
  })
  sr.reveal('.title_1',{origin:'left'})
  sr.reveal('.title_2',{origin:'right'})
  sr.reveal('.vid',{origin:'left'},{delay:500})
  sr.reveal('.summary',{origin:'right'},{delay:500})

function clear(){
  document.getElementById('myForm').reset()
  console.log(12)
}



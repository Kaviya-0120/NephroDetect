/* NephroDetect — Shared UI Scripts */

document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    initNavbarScroll();
    initCustomInputs();
    initFormReset();
});

function initParticles() {
    const canvas = document.getElementById("particles");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let particles = [];
    let animId;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticles() {
        particles = [];
        const count = Math.min(80, Math.floor(window.innerWidth / 15));
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() * 2 + 0.5,
                dx: (Math.random() - 0.5) * 0.4,
                dy: (Math.random() - 0.5) * 0.4,
                opacity: Math.random() * 0.5 + 0.1,
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach((p, i) => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 212, 255, ${p.opacity})`;
            ctx.fill();

            p.x += p.dx;
            p.y += p.dy;
            if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.dy *= -1;

            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(0, 212, 255, ${0.08 * (1 - dist / 120)})`;
                    ctx.stroke();
                }
            }
        });
        animId = requestAnimationFrame(draw);
    }

    resize();
    createParticles();
    draw();
    window.addEventListener("resize", () => { resize(); createParticles(); });
}

function initNavbarScroll() {
    const nav = document.querySelector(".navbar-premium");
    if (!nav) return;
    window.addEventListener("scroll", () => {
        nav.classList.toggle("scrolled", window.scrollY > 40);
    });
}

function initCustomInputs() {
    document.querySelectorAll("[data-custom-trigger]").forEach((select) => {
        const fieldId = select.dataset.customTrigger;
        const wrap = document.getElementById(`${fieldId}-custom-wrap`);
        const customInput = document.getElementById(`${fieldId}-custom`);
        const hiddenInput = document.getElementById(fieldId);

        if (!wrap || !customInput || !hiddenInput) return;

        function updateField() {
            const val = select.value;
            if (val === "custom") {
                wrap.classList.add("show");
                customInput.required = true;
                hiddenInput.value = customInput.value || "";
            } else {
                wrap.classList.remove("show");
                customInput.required = false;
                hiddenInput.value = val;
            }
        }

        select.addEventListener("change", updateField);
        customInput.addEventListener("input", () => {
            if (select.value === "custom") {
                hiddenInput.value = customInput.value;
            }
        });
        updateField();
    });

    const form = document.getElementById("screeningForm");
    if (form) {
        form.addEventListener("submit", (e) => {
            document.querySelectorAll("[data-custom-trigger]").forEach((select) => {
                const fieldId = select.dataset.customTrigger;
                const hiddenInput = document.getElementById(fieldId);
                if (select.value === "custom") {
                    const customInput = document.getElementById(`${fieldId}-custom`);
                    hiddenInput.value = customInput.value;
                } else {
                    hiddenInput.value = select.value;
                }
            });

            const required = ["age", "bp", "sg", "al", "su", "sugar", "bgr", "sc", "hemo"];
            for (const id of required) {
                const el = document.getElementById(id);
                if (!el || !el.value) {
                    e.preventDefault();
                    alert("Please complete all screening fields.");
                    return;
                }
            }
        });
    }
}

function initFormReset() {
    const form = document.getElementById("screeningForm");
    if (!form) return;
    form.addEventListener("reset", () => {
        setTimeout(() => {
            document.querySelectorAll(".custom-input-wrap").forEach((w) => w.classList.remove("show"));
            document.querySelectorAll("[data-custom-trigger]").forEach((select) => {
                select.dispatchEvent(new Event("change"));
            });
        }, 10);
    });
}

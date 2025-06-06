<?php
/* Template Name: WhaleChat */
?>

<!DOCTYPE html>
<html>
<head>
    <title>WhaleChat | VisitWhale</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;700&family=Afacad&display=swap" rel="stylesheet">
    <?php wp_head(); ?>
    <style>
        /* DEIN GESAMTER CSS-STYLE (aus deinem HTML) kommt hier */
        :root {
            --primary: #1a5f8b;
            --secondary: #264653;
            --chat-bg: #212bca;
            --button-color: #bfd3ff;
            --light-blue-bg: #d7e9f7;
            --turquoise: #212bca;
        }

        body {
            font-family: 'Afacad', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fff;
            color: #000;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            background-color: var(--light-blue-bg);
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 0px 0px rgba(0,0,0,0.1);
            width: 98%;
            max-width: 1400px;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }

        .header {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            margin-bottom: 40px;
            text-align: left;
            width: 100%;
        }

        .main-heading-container {
            position: relative;
            width: 100%;
        }

        .main-heading-container::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 0;
            width: 150px;
            height: 5px;
            background-color: var(--turquoise);
        }

        .main-heading {
            font-family: 'Playfair Display', serif;
            font-size: 5.5rem;
            font-weight: 300;
            line-height: 1.1;
            margin: 0 0 10px 0;
            color: var(--turquoise);
            max-width: 800px;
        }

        .subheading {
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            color: var(--turquoise);
            margin: 0;
            font-weight: 300;
            max-width: 600px;
            text-align: left;
        }

        .chat-container {
            margin-top: 100px;
            width: 100%;
            display: flex;
            justify-content: flex-start;
        }

        #chat-form {
            display: flex;
            width: 77%;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            background-color: var(--chat-bg);
            height: 70px;
            align-items: center;
            padding-right: 10px;
        }

        #user_input {
            flex: 1;
            padding: 20px 25px;
            border: none;
            font-size: 1.4rem;
            outline: none;
            color: white;
            background-color: transparent;
            margin-right: 10px;
        }

        #chat-form button {
            background-color: var(--button-color);
            color: black;
            border: none;
            padding: 15px 20px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.2s;
            width: auto;
            min-width: 80px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        #chat-form button:hover {
            background-color: #4c8a93;
        }

        #chat-form button svg {
            width: 30px;
            height: 30px;
            transition: transform 0.2s ease;
        }

        #results {
            margin-top: 60px;
            display: grid;
            grid-template-columns: repeat(2, minmax(550px, 1fr));
            gap: 30px;
        }

        .card-wrapper {
            background: white;
            border-radius: 20px;
            padding: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s;
        }

        .card-wrapper:hover {
            transform: translateY(-3px);
        }

        .property-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: transform 0.3s;
        }

        .property-image {
            width: 100%;
            aspect-ratio: 4 / 3;
            position: relative;
            overflow: hidden;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }

        .property-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .property-info {
            padding: 20px;
        }

        .property-price {
            color: var(--primary);
            font-weight: bold;
            font-size: 1.2rem;
            margin: 10px 0;
        }

        .start-here-btn {
            background-color: var(--button-color);
            color: black;
            border: none;
            padding: 10px 20px;
            font-size: 1.1rem;
            cursor: pointer;
            border-radius: 8px;
            margin-top: 10px;
            transition: background-color 0.2s;
        }

        .start-here-btn:hover {
            background-color: #4c8a93;
        }

        .no-results {
            grid-column: 1 / -1;
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-top: 40px;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
                width: 98%;
            }

            .main-heading {
                font-size: 3rem;
            }

            .subheading {
                font-size: 1.3rem;
            }

            #chat-form {
                width: 100%;
                height: 60px;
            }

            #user_input {
                font-size: 1.2rem;
                padding: 15px 20px;
            }

            #chat-form button {
                padding: 15px 20px;
                font-size: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="main-heading-container">
                <h1 class="main-heading">Less to Decide.<br>More to Travel.</h1>
            </div>
            <p class="subheading">Inspiration or precise request? We find it quickly.</p>
        </div>

        <div class="chat-container">
            <form id="chat-form">
                <input type="text" id="user_input" placeholder="Ready to plan? Start typing..." />
                <button type="submit">
                    <span>Explore</span>
                    <svg height="30" width="30" viewBox="0 0 512 512" fill="#000000" transform="matrix(-1, 0, 0, 1, 0, 0)">
                        <path d="M507.068,194.059c-5.3-6.143-13.759-8.507-21.481-6.013l-59.859,17.264..."/>
                    </svg>
                </button>
            </form>
        </div>

        <div id="results"></div>
    </div>

    <script>
        function animatePrice(element, duration = 2000) {
            const target = parseFloat(element.getAttribute("data-target"));
            let startTime = null;
            function updatePrice(timestamp) {
                if (!startTime) startTime = timestamp;
                const progress = timestamp - startTime;
                const current = Math.min(Math.floor((progress / duration) * target), target);
                element.textContent = `${current} €/night`;
                if (progress < duration) {
                    requestAnimationFrame(updatePrice);
                }
            }
            requestAnimationFrame(updatePrice);
        }

        document.getElementById("chat-form").addEventListener("submit", async function(e) {
            e.preventDefault();
            const userInput = document.getElementById("user_input").value;
            const button = this.querySelector('button');

            button.disabled = true;
            button.innerHTML = '<span>Exploring...</span>';

            try {
                const res = await fetch("/whalechat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: userInput })
                });

                const data = await res.json();
                const resultsDiv = document.getElementById("results");

                if(data.results.length === 0) {
                    resultsDiv.innerHTML = `<div class="no-results">No perfect matches found. Try different keywords!</div>`;
                } else {
                    resultsDiv.innerHTML = data.results.map(r => `
                        <div class="card-wrapper">
                            <div class="property-card">
                                <div class="property-image">
                                    <img src="${r.image_url}">
                                </div>
                                <div class="property-info">
                                    <h3>${r.name}</h3>
                                    <div>${r.city}, ${r.country}</div>
                                    <div class="property-price" data-target="${r.price}">0 €/night</div>
                                    <button class="start-here-btn" onclick="window.open('${r.link}', '_blank')">Start Here</button>
                                </div>
                            </div>
                        </div>
                    `).join("");

                    document.querySelectorAll(".property-price").forEach(el => {
                        animatePrice(el);
                    });
                }
            } catch(error) {
                console.error(error);
                alert("Search failed. Please try again.");
            } finally {
                button.disabled = false;
                button.innerHTML = 'Explore';
            }
        });
    </script>
    <?php wp_footer(); ?>
</body>
</html>

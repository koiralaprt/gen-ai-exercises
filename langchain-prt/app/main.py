from app.extraction import extract_company
from app.agent import agent_executor

essay = """
In the ever-evolving landscape of global commerce, the origin stories of
major corporations are not merely tales of personal ambition and
entrepreneurial spirit but also reflections of broader socio-economic
trends and technological revolutions that have reshaped industries. These
narratives, which often begin with modest ambitions, unfold into chronicles
of innovation and strategic foresight that define industries and set
benchmarks for future enterprises.
Early Foundations: Pioneers of Industry
One of the earliest examples is The Coca-Cola Company, founded on May 8,
1886, by Dr. John Stith Pemberton in Atlanta, Georgia. Initially sold at
Jacob's Pharmacy as a medicinal beverage, Coca-Cola would become one of the
most recognized brands worldwide, revolutionizing the beverage industry.
Similarly, Sony Corporation was established on May 7, 1946, by Masaru Ibuka
and Akio Morita in Tokyo, Japan. Starting with repairing and building
electrical equipment in post-war Japan, Sony would grow to pioneer
electronics, entertainment, and technology.
As the mid-20th century progressed, McDonald’s Corporation emerged as a
game-changer in the fast-food industry. Founded on April 15, 1955, in Des
Plaines, Illinois, by Ray Kroc, McDonald’s built upon the original concept
of Richard and Maurice McDonald to standardize and scale fast-food service
globally. Around the same period, Intel Corporation was established on July
18, 1968, by Robert Noyce and Gordon Moore in Mountain View, California,

driving advancements in semiconductors and microprocessors that became the
backbone of modern computing.
The Rise of Technology Titans
Samsung Electronics Co., Ltd., founded on January 13, 1969, by Lee
Byung-chul in Su-dong, South Korea, initially focused on producing
electrical appliances like televisions and refrigerators. As Samsung
expanded into semiconductors, telecommunications, and digital media, it
grew into a global technology leader. Similarly, Microsoft Corporation was
founded on April 4, 1975, by Bill Gates and Paul Allen in Albuquerque, New
Mexico, with the vision of placing a computer on every desk and in every
home.
In Cupertino, California, Apple Inc. was born on April 1, 1976, founded by
Steve Jobs, Steve Wozniak, and Ronald Wayne. Their mission to make personal
computing accessible and elegant revolutionized technology and design. A
few years later, Oracle Corporation was established on June 16, 1977, by
Larry Ellison, Bob Miner, and Ed Oates in Santa Clara, California.
Specializing in relational databases, Oracle would become a cornerstone of
enterprise software and cloud computing.
NVIDIA Corporation, founded on April 5, 1993, by Jensen Huang, Chris
Malachowsky, and Curtis Priem in Santa Clara, California, began with a
focus on graphics processing units (GPUs) for gaming. Today, NVIDIA is a
leader in artificial intelligence, deep learning, and autonomous systems,
showcasing the power of continuous innovation.
E-Commerce and the Internet Revolution
The 1990s witnessed a dramatic shift toward e-commerce and internet
technologies. Amazon.com Inc. was founded on July 5, 1994, by Jeff Bezos in
a garage in Bellevue, Washington, with the vision of becoming the world’s
largest online bookstore. This vision rapidly expanded to encompass
e-commerce, cloud computing, and digital streaming. Similarly, Google LLC
was founded on September 4, 1998, by Larry Page and Sergey Brin, PhD
students at Stanford University, in a garage in Menlo Park, California.
Google’s mission to “organize the world’s information” transformed how we
search, learn, and connect.
In Asia, Alibaba Group Holding Limited was founded on June 28, 1999, by
Jack Ma and 18 colleagues in Hangzhou, China. Originally an e-commerce
platform connecting manufacturers with buyers, Alibaba expanded into cloud

computing, digital entertainment, and financial technology, becoming a
global powerhouse.
In Europe, SAP SE was founded on April 1, 1972, by Dietmar Hopp,
Hans-Werner Hector, Hasso Plattner, Klaus Tschira, and Claus Wellenreuther
in Weinheim, Germany. Specializing in enterprise resource planning (ERP)
software, SAP revolutionized how businesses manage operations and data.
Social Media and Digital Platforms
The 2000s brought a wave of social media and digital platforms that
reshaped communication and commerce. LinkedIn Corporation was founded on
December 28, 2002, by Reid Hoffman and a team from PayPal and Socialnet.com
in Mountain View, California, focusing on professional networking.
Facebook, Inc. (now Meta Platforms, Inc.) was launched on February 4, 2004,
by Mark Zuckerberg and his college roommates in Cambridge, Massachusetts,
evolving into a global social networking behemoth.
Another transformative platform, Twitter, Inc., was founded on March 21,
2006, by Jack Dorsey, Biz Stone, and Evan Williams in San Francisco,
California. Starting as a microblogging service, Twitter became a critical
tool for communication and social commentary. Spotify AB, founded on April
23, 2006, by Daniel Ek and Martin Lorentzon in Stockholm, Sweden, leveraged
streaming technology to democratize music consumption, fundamentally
altering the music industry.
In the realm of video-sharing, YouTube LLC was founded on February 14,
2005, by Steve Chen, Chad Hurley, and Jawed Karim in San Mateo, California.
YouTube became the leading platform for user-generated video content,
influencing global culture and media consumption.
Innovators in Modern Technology
Tesla, Inc., founded on July 1, 2003, by a group including Elon Musk,
Martin Eberhard, Marc Tarpenning, JB Straubel, and Ian Wright, in San
Carlos, California, championed the transition to sustainable energy with
its electric vehicles and energy solutions. Airbnb, Inc., founded in August
2008 by Brian Chesky, Joe Gebbia, and Nathan Blecharczyk in San Francisco,
California, disrupted traditional hospitality with its peer-to-peer lodging
platform.
In the realm of fintech, PayPal Holdings, Inc. was established in December
1998 by Peter Thiel, Max Levchin, Luke Nosek, and Ken Howery in Palo Alto,

California. Originally a cryptography company, PayPal became a global
leader in online payments. Stripe, Inc., founded in 2010 by Patrick and
John Collison in Palo Alto, California, followed suit, simplifying online
payments and enabling digital commerce.
Square, Inc. (now Block, Inc.), founded on February 20, 2009, by Jack
Dorsey and Jim McKelvey in San Francisco, California, revolutionized mobile
payment systems with its simple and accessible card readers.
Recent Disruptors
Zoom Video Communications, Inc. was founded on April 21, 2011, by Eric Yuan
in San Jose, California. Initially designed for video conferencing, Zoom
became essential during the COVID-19 pandemic, transforming remote work and
communication. Slack Technologies, LLC, founded in 2009 by Stewart
Butterfield, Eric Costello, Cal Henderson, and Serguei Mourachov in
Vancouver, Canada, redefined workplace communication with its innovative
messaging platform.
Rivian Automotive, Inc., founded on June 23, 2009, by RJ Scaringe in
Plymouth, Michigan, entered the electric vehicle market with a focus on
adventure and sustainability. SpaceX, established on March 14, 2002, by
Elon Musk in Hawthorne, California, revolutionized aerospace with reusable
rockets and ambitious plans for Mars exploration.
TikTok, developed by ByteDance and launched in September 2016 by Zhang
Yiming in Beijing, China, revolutionized short-form video content, becoming
a cultural phenomenon worldwide.
Conclusion
These corporations, with their diverse beginnings and visionary founders,
exemplify the interplay of innovation, timing, and strategic foresight that
shapes industries and transforms markets. From repairing electronics in
post-war Japan to building global e-commerce empires and redefining space
exploration, their stories are milestones in the narrative of global
economic transformation. Each reflects not only the aspirations of their
founders but also the technological advancements and socio-economic trends
of their time, serving as inspirations for future innovators.
"""

# Simple split to handle paragraphs
paragraphs = [p.strip() for p in essay.strip().split('\n\n') if p.strip()]

for i, paragraph in enumerate(paragraphs):
    try:
        company_data = extract_company(paragraph)
        print(f"company {company_data.company_name} (Founded: {company_data.founding_date})")
    except Exception as e:
        print(f" Error: {e}")
        continue 

    data_dict = company_data.model_dump()
    
    agent_instruction = (
        f"**URGENT INSTRUCTION:** You have been provided with the following validated, structured company data. "
        f"**YOU MUST USE THE 'db_insert_tool' TOOL** to save this data and complete the task. "
        f"The details for insertion are: "
        f"company_name: {data_dict['company_name']}, "
        f"founding_date: {data_dict['founding_date']}, "
        f"founders: {', '.join(data_dict['founders'])}"
    )
    try:
        result = agent_executor.invoke({"messages": agent_instruction})
        print("\n Agent execution completed.")
        final_output = result.get('output', 'completed.')
        print(f"\n res:{final_output}")
    except Exception as e:
        print("error during agent execution:")
        print(f"\n err: {e}")
        